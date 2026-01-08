#include "lfcalibrate.h"

#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "json.hpp"
#include "lfio.h"
#include "utils.h"

#include <format>
#include <numeric> // for std::iota
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using json = nlohmann::json;

LFCalibrate::LFCalibrate(const cv::Mat &white_img) { setImage(white_img); }

void LFCalibrate::setImage(const cv::Mat &img) {
	if (img.empty()) {
		throw std::runtime_error("LFCalibrate: Input image is empty!");
	}
	if (img.channels() != 1) {
		// 自动转灰度，增加易用性
		cv::cvtColor(img, _white_img, cv::COLOR_BGR2GRAY);
	} else {
		_white_img = img.clone();
	}
}

void LFCalibrate::initConfigLytro2() {
	config.bayer = BayerPattern::GRBG;
	config.bitDepth = 10;
}

std::vector<std::vector<cv::Point2f>> LFCalibrate::run() {
	if (_white_img.empty()) {
		throw std::runtime_error(
			"LFCalibrate: No image set. Call setImage() first.");
	}

	// -------------------------------------------------------------------------
	// 1. 预处理：消除 Bayer 棋盘格
	// -------------------------------------------------------------------------
	if (config.bayer != BayerPattern::NONE) {
		// 使用高斯模糊平滑 Bayer 纹理，保留几何质心
		cv::GaussianBlur(_white_img, _white_img, cv::Size(3, 3), 0);
	}

	// -------------------------------------------------------------------------
	// 2. 预处理：位深归一化 (转 8-bit)
	// -------------------------------------------------------------------------
	if (_white_img.depth() != CV_8U) {
		double scale = 1.0;
		if (config.bitDepth > 8) {
			scale = 255.0 / ((1 << config.bitDepth) - 1);
		} else if (_white_img.depth() == CV_16U) {
			scale = 255.0 / 65535.0;
		}
		_white_img.convertTo(_white_img, CV_8U, scale);
	}

	// -------------------------------------------------------------------------
	// 3. 质心提取 (Centroids Extract)
	// -------------------------------------------------------------------------
	CentroidsExtract ce(_white_img);
	if (config.autoEstimate) {
		ce.run(config.ceMethod);
		config.diameter = ce.getEstimatedM();
	} else {
		ce.run(config.ceMethod, config.diameter);
	}

	// -------------------------------------------------------------------------
	// 4. 排序与网格化 (Centroids Sort)
	// -------------------------------------------------------------------------
	CentroidsSort cs(ce.getPoints(), ce.getPitch());
	cs.run2(); // 使用泛洪填充算法

	// [关键修改] 获取并保存奇偶行相位
	_hex_odd = cs.getHexOdd();

	// -------------------------------------------------------------------------
	// 5. 网格拟合 (HexGrid Fit)
	// -------------------------------------------------------------------------
	HexGridFitter hgf(cs.getPoints(), cs.getPointsSize(), _hex_odd);

	// 使用快速鲁棒拟合
	hgf.fitFastRobust(2.0f, 1500);

	_points = hgf.predict();

	return _points;
}

void LFCalibrate::savePoints(const std::string &filename) {
	if (_points.empty())
		return;

	json j;
	j["rows"] = static_cast<int>(_points.size());
	j["cols"] = static_cast<int>(_points[0].size());

	std::vector<cv::Point2f> flat_data;
	flat_data.reserve(_points.size() * _points[0].size());

	for (const auto &row : _points) {
		flat_data.insert(flat_data.end(), row.begin(), row.end());
	}

	j["data"] = flat_data;
	writeJson(filename, j);
}

// ---------------------------------------------------------
// Map Computation Workers
// ---------------------------------------------------------

void LFCalibrate::_computeExtractMaps(int winSize) {
	if (_points.empty() || _points[0].empty()) {
		std::cerr
			<< "[LFCalibrate] Error: No points data. Run calibration first."
			<< std::endl;
		return;
	}

	int m_rows = _points.size();
	int m_cols = _points[0].size();
	int total_views = winSize * winSize;

	_extract_maps.clear();
	_extract_maps.resize(total_views * 2);

	float startOffset = -(winSize - 1) / 2.0f;

#pragma omp parallel for
	for (int u = 0; u < winSize; ++u) {
		for (int v = 0; v < winSize; ++v) {
			float off_y = startOffset + u;
			float off_x = startOffset + v;

			cv::Mat map_x(m_rows, m_cols, CV_32FC1);
			cv::Mat map_y(m_rows, m_cols, CV_32FC1);

			for (int r = 0; r < m_rows; ++r) {
				float *ptr_x = map_x.ptr<float>(r);
				float *ptr_y = map_y.ptr<float>(r);
				const auto &row_points = _points[r];

				for (int c = 0; c < m_cols; ++c) {
					ptr_x[c] = row_points[c].x + off_x;
					ptr_y[c] = row_points[c].y + off_y;
				}
			}

			int idx = (u * winSize + v) * 2;
			_extract_maps[idx] = map_x;
			_extract_maps[idx + 1] = map_y;
		}
	}
}

void LFCalibrate::_computeDehexMaps() {
	if (_extract_maps.empty()) {
		std::cerr
			<< "[LFCalibrate] Error: Extract maps empty. Compute them first."
			<< std::endl;
		return;
	}

	// 1. 获取网格尺寸
	int grid_rows = _extract_maps[0].rows;
	int grid_cols = _extract_maps[0].cols;

	// 2. 计算反向拉伸参数
	int dstCols = std::round(2.0 * grid_cols / std::sqrt(3.0));
	double scale = (double)grid_cols / dstCols;

	// 3. 准备变换向量
	// Map Y: 简单的行复制
	cv::Mat col_vec(grid_rows, 1, CV_32FC1);
	std::iota(col_vec.begin<float>(), col_vec.end<float>(), 0.0f);

	cv::Mat map_y;
	cv::repeat(col_vec, 1, dstCols, map_y);

	// Map X: 基础线性增长
	cv::Mat row_vec(1, dstCols, CV_32FC1);
	std::iota(row_vec.begin<float>(), row_vec.end<float>(), 0.0f);
	row_vec *= scale; // 拉伸

	cv::Mat base_x;
	cv::repeat(row_vec, grid_rows, 1, base_x);

	// 4. [核心修改] 根据 _hex_odd 决定偏移方向
	// row_indices & 1 产生 0, 1, 0, 1... 序列
	cv::Mat row_indices(grid_rows, 1, CV_32S);
	std::iota(row_indices.begin<int>(), row_indices.end<int>(), 0);

	// 确定偏移量：如果 hex_odd
	// 为真，说明奇数行原本向右偏了0.5，为了对齐我们需要减去0.5 如果 hex_odd
	// 为假，说明奇数行向左偏了0.5，我们需要加0.5 (即减去 -0.5)
	float shift_val = _hex_odd ? -0.5f : 0.5f;

	cv::Mat shift_vec;
	// 只有奇数行会有值 shift_val，偶数行为 0
	cv::Mat(row_indices & 1).convertTo(shift_vec, CV_32F, shift_val);

	cv::Mat shift_map;
	cv::repeat(shift_vec, 1, dstCols, shift_map);

	// 应用偏移
	cv::Mat map_x = base_x - shift_map;

	// 5. 缓存结果
	_dehex_maps.clear();
	_dehex_maps.reserve(2);
	_dehex_maps.push_back(map_x);
	_dehex_maps.push_back(map_y);
}

// ---------------------------------------------------------
// Public Interface Implementation
// ---------------------------------------------------------

const std::vector<cv::Mat> &LFCalibrate::computeExtractMaps(int winSize) {
	_computeExtractMaps(winSize);
	std::cout << std::format(
		"[LFCalibrate] Extract maps computed. (map shape: {}x{}x{}x{})",
		winSize, winSize, _extract_maps[0].cols, _extract_maps[0].rows)
			  << std::endl;
	return _extract_maps;
}

void LFCalibrate::getExtractMaps(cv::Mat &out_x, cv::Mat &out_y, int row,
								 int col) const {
	if (_extract_maps.empty()) {
		std::cerr << "[LFCalibrate] Error: Maps not computed." << std::endl;
		return;
	}

	int total_maps = _extract_maps.size();
	int winSize = std::sqrt(total_maps / 2);
	int u = (row == -1) ? winSize / 2 : row;
	int v = (col == -1) ? winSize / 2 : col;

	if (u < 0 || u >= winSize || v < 0 || v >= winSize)
		return;

	int idx = (u * winSize + v) * 2;
	out_x = _extract_maps[idx];
	out_y = _extract_maps[idx + 1];
}

const std::vector<cv::Mat> &LFCalibrate::computeDehexMaps() {
	_computeDehexMaps();
	std::cout << std::format(
		"[LFCalibrate] Dehex maps computed (HexOdd: {}, Shape: {}x{}).",
		_hex_odd ? "True" : "False", _dehex_maps[0].cols, _dehex_maps[0].rows)
			  << std::endl;
	return _dehex_maps;
}

void LFCalibrate::getDehexMaps(cv::Mat &out_x, cv::Mat &out_y) {
	if (_dehex_maps.empty()) {
		_computeDehexMaps();
	}
	if (!_dehex_maps.empty()) {
		out_x = _dehex_maps[0];
		out_y = _dehex_maps[1];
	}
}