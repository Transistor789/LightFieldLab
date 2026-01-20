#include "lfcalibrate.h"

#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "json.hpp"
#include "utils.h"

#include <format>
#include <numeric> // for std::iota
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

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

void LFCalibrate::run(const cv::Mat &img, const CalibrateConfig &config) {
	if (img.empty()) {
		throw std::runtime_error("LFCalibrate: No image set. Call setImage() first.");
	}

	cv::Mat temp;
	if (config.orientation == Orientation::VERT) {
		cv::transpose(img, temp);
	} else {
		temp = img;
	}

	// 1. 预处理：消除 Bayer 棋盘格
	if (config.bayer != BayerPattern::NONE) {
		// 使用高斯模糊平滑 Bayer 纹理，保留几何质心
		cv::GaussianBlur(temp, temp, cv::Size(3, 3), 0);
	}

	// 2. 预处理：位深归一化 (转 8-bit)
	if (temp.depth() != CV_8U) {
		double scale = 1.0;
		if (config.bitDepth > 8) {
			scale = 255.0 / ((1 << config.bitDepth) - 1);
		} else if (temp.depth() == CV_16U) {
			scale = 255.0 / 65535.0;
		}
		temp.convertTo(temp, CV_8U, scale);
	}

	// 3. 质心提取 (Centroids Extract)
	CentroidsExtract ce(temp);
	if (config.autoEstimate) {
		ce.run(config.ceMethod);
		_diameter = ce.getEstimatedM();
	} else {
		ce.run(config.ceMethod, config.diameter);
		_diameter = config.diameter;
	}

	// 4. 排序与网格化 (Centroids Sort)
	CentroidsSort cs(ce.getPoints(), ce.getPitch());
	cs.run2(); // 使用泛洪填充算法
	_hex_odd = cs.getHexOdd();

	if (config.hexgridfit) {
		// 5. 网格拟合 (HexGrid Fit)
		_fitter = std::make_shared<HexGridFitter>(cs.getPointsAsMats(), _hex_odd);

		// 使用快速鲁棒拟合
		_fitter->fitFastRobust(2.0f, 1500);
		_maps = _fitter->predict();
	} else {
		_maps = cs.getPointsAsMats();
	}

	if (config.genLUT) {
		computeExtractMaps(config.views, config.space);
		computeDehexMaps();
		if (config.orientation == Orientation::VERT) {
			cv::transpose(_maps.first, _maps.first);
			cv::transpose(_maps.second, _maps.second);
			std::swap(_maps.first, _maps.second);

			for (size_t i = 0; i < _extract_maps.size(); i += 2) {
				cv::transpose(_extract_maps[i], _extract_maps[i]);
				cv::transpose(_extract_maps[i + 1], _extract_maps[i + 1]);
				std::swap(_extract_maps[i], _extract_maps[i + 1]);
			}

			for (size_t i = 0; i < _dehex_maps.size(); i += 2) {
				cv::transpose(_dehex_maps[i], _dehex_maps[i]);
				cv::transpose(_dehex_maps[i + 1], _dehex_maps[i + 1]);
				std::swap(_dehex_maps[i], _dehex_maps[i + 1]);
			}
		}
	}
}

void LFCalibrate::savePoints(const std::string &filename) {
	// 1. 安全检查：确保坐标矩阵不为空
	if (_maps.first.empty() || _maps.second.empty())
		return;

	// 2. 获取网格尺寸
	int rows = _maps.first.rows;
	int cols = _maps.first.cols;

	json j;
	j["rows"] = rows;
	j["cols"] = cols;

	// 3. 准备平坦化的数据容器
	std::vector<cv::Point2f> flat_data;
	flat_data.reserve(rows * cols);

	// 4. 遍历矩阵并将 X/Y 重新组合为 Point2f 存入 vector
	for (int r = 0; r < rows; ++r) {
		const float *ptr_x = _maps.first.ptr<float>(r);
		const float *ptr_y = _maps.second.ptr<float>(r);

		for (int c = 0; c < cols; ++c) {
			// 将分开存储的 x 和 y 重新封装为点对象
			flat_data.emplace_back(ptr_x[c], ptr_y[c]);
		}
	}

	// 5. 序列化并写入文件
	j["data"] = flat_data;
	writeJson(filename, j);
}

const std::vector<cv::Mat> &LFCalibrate::computeExtractMaps(int winSize, float space) {
	// 1. 安全检查：检查 X 和 Y 坐标矩阵是否为空
	if (_maps.first.empty() || _maps.second.empty()) {
		std::cerr << "[LFCalibrate] Error: No maps data. Run centroids sort first." << std::endl;
		return _extract_maps;
	}

	int m_rows = _maps.first.rows;
	int m_cols = _maps.first.cols;
	int total_views = winSize * winSize;

	// 2. 初始化查找表容器
	_extract_maps.clear();
	_extract_maps.resize(total_views * 2);

	// 3. 计算视图偏移基准
	float startOffset = -(winSize - 1) / 2.0f;

	// 4. 并行计算查找表
#pragma omp parallel for
	for (int u = 0; u < winSize; ++u) {
		for (int v = 0; v < winSize; ++v) {
			// 计算当前视角的偏移量
			float off_y = (startOffset + u) * space;
			float off_x = (startOffset + v) * space;

			// 5. 利用 OpenCV 矩阵算术运算简化代码
			// 直接将偏移量加到中心点矩阵上，生成当前视角的采样 Map
			cv::Mat map_x = _maps.first + off_x;
			cv::Mat map_y = _maps.second + off_y;

			int idx = (u * winSize + v) * 2;
			_extract_maps[idx] = map_x;
			_extract_maps[idx + 1] = map_y;
		}
	}

	// 6. 打印日志
	std::cout << std::format(
		"[LFCalibrate] Extract maps computed from Mats. (winSize: {}, space: {:.2f}, "
		"map: {}x{})",
		winSize, space, m_cols, m_rows)
			  << std::endl;

	return _extract_maps;
}

const std::vector<cv::Mat> &LFCalibrate::computeDehexMaps() {
	// 1. 安全检查：确保提取映射已存在
	if (_extract_maps.empty()) {
		std::cerr << "[LFCalibrate] Error: Extract maps empty. Compute them first." << std::endl;
		return _dehex_maps;
	}

	// 2. 获取网格尺寸并计算拉伸参数
	int grid_rows = _extract_maps[0].rows;
	int grid_cols = _extract_maps[0].cols;

	// 计算反向拉伸：由于六边形网格在 X 方向上被压缩了
	// sqrt(3)/2，这里需要扩回矩形
	int dstCols = std::round(2.0 * grid_cols / std::sqrt(3.0));
	double scale = (double)grid_cols / dstCols;

	// 3. 准备映射矩阵 (Map Y: 简单的行复制)
	cv::Mat col_vec(grid_rows, 1, CV_32FC1);
	std::iota(col_vec.begin<float>(), col_vec.end<float>(), 0.0f);

	cv::Mat map_y;
	cv::repeat(col_vec, 1, dstCols, map_y);

	// 4. 准备 Map X (基础线性拉伸 + 奇偶行偏移校正)
	cv::Mat row_vec(1, dstCols, CV_32FC1);
	std::iota(row_vec.begin<float>(), row_vec.end<float>(), 0.0f);
	row_vec *= scale;

	cv::Mat base_x;
	cv::repeat(row_vec, grid_rows, 1, base_x);

	// 处理六边形偏移：决定偏移量 shift_val
	// HexOdd 为真说明奇数行原本偏右 0.5，对齐需减 0.5；反之加 0.5
	float shift_val = _hex_odd ? -0.5f : 0.5f;

	cv::Mat shift_vec(grid_rows, 1, CV_32FC1);
	for (int i = 0; i < grid_rows; ++i) {
		shift_vec.at<float>(i) = (i % 2 == 0) ? 0.0f : shift_val;
	}

	cv::Mat shift_map;
	cv::repeat(shift_vec, 1, dstCols, shift_map);

	// 应用偏移计算最终 map_x
	cv::Mat map_x = base_x - shift_map;

	// 5. 缓存结果并打印日志
	_dehex_maps.clear();
	_dehex_maps.reserve(2);
	_dehex_maps.push_back(map_x);
	_dehex_maps.push_back(map_y);

	std::cout << std::format("[LFCalibrate] Dehex maps computed (HexOdd: {}, Shape: {}x{}).",
							 _hex_odd ? "True" : "False", dstCols, grid_rows)
			  << std::endl;

	return _dehex_maps;
}

std::vector<cv::Mat> LFCalibrate::getExtractMaps() const {
	if (_extract_maps.empty()) {
		std::cerr << "[LFCalibrate] Error: Maps not computed." << std::endl;
		return {};
	}

	return _extract_maps;
}

std::vector<cv::Mat> LFCalibrate::getDehexMaps() const {
	if (_dehex_maps.empty()) {
		std::cerr << "[LFCalibrate] Error: LUT not computed." << std::endl;
		return {};
	}
	return _dehex_maps;
}
