#include "lfcalibrate.h"

#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "json.hpp"
#include "lfio.h"
#include "utils.h"

#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using json = nlohmann::json;

LFCalibrate::LFCalibrate(const cv::Mat &white_img) { setImage(white_img); }

void LFCalibrate::setImage(const cv::Mat &img) {
	if (img.channels() != 1) {
		throw std::runtime_error("Calibrate requires single channel image!");
	}
	_white_img = img.clone();
}

void LFCalibrate::initConfigLytro2() {
	config.bayer = BayerPattern::GRBG;
	config.bitDepth = 10;
}

std::vector<std::vector<cv::Point2f>> LFCalibrate::run() {
	if (config.bayer != BayerPattern::NONE) {
		cv::Mat temp;
		cv::demosaicing(_white_img, temp,
						get_demosaic_code(config.bayer, true));
		_white_img = temp;
	}

	if (_white_img.type() != CV_8U || _white_img.type() != CV_8UC1) {
		if (config.bitDepth == 8) {
			throw std::runtime_error(std::string(__FUNCTION__)
									 + "白图像位深参数错误");
		}
		_white_img.convertTo(_white_img, CV_8U,
							 255.0 / ((1 << config.bitDepth) - 1));
	}

	CentroidsExtract ce(_white_img);
	if (config.autoEstimate) {
		ce.run(config.use_cca);
		config.diameter = ce.getEstimatedM();
	} else {
		ce.run(config.use_cca, config.diameter);
	}

	CentroidsSort cs(ce.getPoints(), ce.getPitch());
	cs.run();

	HexGridFitter hgf(cs.getPoints(), cs.getPointsSize(), cs.getHexOdd());
	hgf.fit();
	_points = hgf.predict();

	return _points;
}

void LFCalibrate::savePoints(const std::string &filename) {
	json j;
	j["rows"] = static_cast<int>(_points.size());
	j["cols"] = _points.empty() ? 0 : static_cast<int>(_points[0].size());

	std::vector<cv::Point2f> flat_data;
	flat_data.reserve(_points.size()
					  * (_points.empty() ? 0 : _points[0].size()));

	for (const auto &row : _points) {
		flat_data.insert(flat_data.end(), row.begin(), row.end());
	}

	j["data"] = flat_data;

	writeJson(filename, j);
}

// ---------------------------------------------------------
// 1. 私有 Worker：负责繁重的计算
// ---------------------------------------------------------
void LFCalibrate::_computeExtractMaps(int winSize) {
	if (_points.empty() || _points[0].empty()) {
		std::cerr << "[LFCalibrate] Error: No points data available."
				  << std::endl;
		return;
	}

	int m_rows = _points.size();
	int m_cols = _points[0].size();
	int total_views = winSize * winSize;

	// 重新分配内存
	_extract_maps.clear();
	_extract_maps.resize(total_views * 2);

	float startOffset = -(winSize - 1) / 2.0f;

// OpenMP 并行计算
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

			int base_idx = (u * winSize + v) * 2;
			_extract_maps[base_idx] = map_x;
			_extract_maps[base_idx + 1] = map_y;
		}
	}
	std::cout << "[LFCalibrate] Extract maps computed and cached." << std::endl;
}

// ---------------------------------------------------------
// 2. 公开接口：强制计算并获取
// ---------------------------------------------------------
void LFCalibrate::computeExtractMaps(int winSize, cv::Mat &out_x,
									 cv::Mat &out_y, int row, int col) {
	_computeExtractMaps(winSize);			// 强制刷新缓存
	getExtractMaps(out_x, out_y, row, col); // 获取指定视角
}

std::vector<cv::Mat> LFCalibrate::computeExtractMaps(int winSize) {
	_computeExtractMaps(winSize); // 强制刷新缓存

	// --- 输出 Extract Maps 尺寸信息 ---
	if (!_extract_maps.empty()) {
		std::cout << "[LFCalibrate] Extract Maps generated. Count: "
				  << _extract_maps.size()
				  << ", Resolution: " << _extract_maps[0].cols << "x"
				  << _extract_maps[0].rows << std::endl;
	} else {
		std::cout << "[LFCalibrate] Warning: Extract Maps are empty."
				  << std::endl;
	}
	// ----------------------------------

	return _extract_maps;
}

// ---------------------------------------------------------
// 3. 公开接口：获取 (Getter)
// ---------------------------------------------------------
void LFCalibrate::getExtractMaps(cv::Mat &out_x, cv::Mat &out_y, int row,
								 int col) {
	// 安全检查：如果没初始化，直接返回
	if (_extract_maps.empty()) {
		std::cerr << "[Calibrate] Error: Extract maps not initialized. Call "
					 "computeExtractMaps first."
				  << std::endl;
		return;
	}

	// 计算 WinSize (反推)
	int total_maps = _extract_maps.size();
	int winSize = std::sqrt(total_maps / 2);

	// 默认取中心
	int u = (row == -1) ? winSize / 2 : row;
	int v = (col == -1) ? winSize / 2 : col;

	// 越界检查
	if (u < 0 || u >= winSize || v < 0 || v >= winSize) {
		// 可以在这里打印 Error
		return;
	}

	int idx = (u * winSize + v) * 2;
	out_x = _extract_maps[idx];
	out_y = _extract_maps[idx + 1];
}

void LFCalibrate::_computeDehexMaps() {
	// 1. 依赖检查：必须先有 Extract Map 才能确定尺寸
	if (_extract_maps.empty()) {
		std::cerr
			<< "[Calibrate] Error: Extract maps are empty. Cannot compute "
			   "Dehex maps."
			<< std::endl;
		return;
	}

	// 2. 从 Extract Map 获取微透镜阵列尺寸
	int grid_rows = _extract_maps[0].rows;
	int grid_cols = _extract_maps[0].cols;

	// 3. 计算 Dehex 逻辑 (反向拉伸)
	int dstCols = std::round(2.0 * grid_cols / std::sqrt(3.0));
	double scale = (double)grid_cols / dstCols;

	cv::Mat map_x, map_y;

	// --- 生成 Map Y ---
	cv::Mat col_vec(grid_rows, 1, CV_32FC1);
	std::iota(col_vec.begin<float>(), col_vec.end<float>(), 0.0f);
	cv::repeat(col_vec, 1, dstCols, map_y);

	// --- 生成 Map X ---
	cv::Mat row_vec(1, dstCols, CV_32FC1);
	std::iota(row_vec.begin<float>(), row_vec.end<float>(), 0.0f);
	row_vec *= scale;

	cv::Mat base_x;
	cv::repeat(row_vec, grid_rows, 1, base_x);

	cv::Mat row_indices(grid_rows, 1, CV_32S);
	std::iota(row_indices.begin<int>(), row_indices.end<int>(), 0);

	cv::Mat shift_vec;
	cv::Mat(row_indices & 1).convertTo(shift_vec, CV_32F, 0.5);

	cv::Mat shift_map;
	cv::repeat(shift_vec, 1, dstCols, shift_map);

	map_x = base_x - shift_map;

	// 4. 存入 Vector 缓存
	_dehex_maps.clear();
	_dehex_maps.reserve(2);
	_dehex_maps.push_back(map_x);
	_dehex_maps.push_back(map_y);

	std::cout << "[LFCalibrate] Dehex maps computed and cached." << std::endl;
}

std::vector<cv::Mat> LFCalibrate::computeDehexMaps() {
	_computeDehexMaps();

	// --- 输出 Dehex Maps 尺寸信息 ---
	if (!_dehex_maps.empty()) {
		std::cout << "[LFCalibrate] Dehex Maps generated. Count: "
				  << _dehex_maps.size()
				  << ", Resolution: " << _dehex_maps[0].cols << "x"
				  << _dehex_maps[0].rows << std::endl;
	} else {
		std::cout << "[LFCalibrate] Warning: Dehex Maps are empty."
				  << std::endl;
	}
	// --------------------------------

	return _dehex_maps;
}

void LFCalibrate::computeDehexMaps(cv::Mat &out_x, cv::Mat &out_y) {
	_computeDehexMaps();
	getDehexMaps(out_x, out_y);
}

void LFCalibrate::getDehexMaps(cv::Mat &out_x, cv::Mat &out_y) {
	if (_dehex_maps.empty()) {
		_computeDehexMaps();
	}

	// 2. 安全检查：如果计算失败 (例如没有 SliceMap)，直接返回
	if (_dehex_maps.empty()) {
		// computeDehexMaps 内部已经打印了 Error 信息，这里无需重复
		return;
	}

	// 3. 赋值返回
	// Dehex Map 与视角 (row, col) 无关，是全局通用的几何变换
	// 直接取出唯一的 X 和 Y 表 (浅拷贝，极快)
	out_x = _dehex_maps[0];
	out_y = _dehex_maps[1];
}