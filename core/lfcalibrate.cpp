#include "lfcalibrate.h"

#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "lfio.h"
#include "utils.h"

#include <json.hpp>

using json = nlohmann::json;

LFCalibrate::LFCalibrate() {}

LFCalibrate::LFCalibrate(const cv::Mat &white_img) { setImage(white_img); }

void LFCalibrate::setImage(const cv::Mat &img) {
	if (img.channels() != 1) {
		throw std::runtime_error("Calibrate requires single channel image!");
	}
	_white_img = img.clone();
}

std::vector<std::vector<cv::Point2f>> LFCalibrate::run(bool use_cca,
													   bool save) {
	CentroidsExtract ce(_white_img);
	ce.run(use_cca);
	std::vector<cv::Point2f> pts = ce.getPoints();
	auto pitch = ce.getPitch();

	CentroidsSort cs(pts, pitch);
	cs.run();
	std::vector<cv::Point2f> pts_sorted = cs.getPoints();
	std::vector<int> size = cs.getPointsSize();
	bool hex_odd = cs.getHexOdd();

	HexGridFitter hgf(pts_sorted, size, hex_odd);
	hgf.fit();
	auto pts_fitted = hgf.predict();
	_points = pts_fitted;

	if (save) {
		if (use_cca) {
			draw_points(_white_img, pts, "../data/centers_detected_cca.png", 1,
						0, true);
		} else {
			draw_points(_white_img, pts, "../data/centers_detected_moments.png",
						1, 0, true);
		}

		draw_points(_white_img, pts_sorted, "../data/centers_sorted.png", 1, 0,
					true);
		draw_points(_white_img, pts_fitted, "../data/centers_fitted.png", 1, 0,
					true);

		savePoints("../data/centroids.json");
	}

	return pts_fitted;
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
void LFCalibrate::_computeSliceMaps(int winSize) {
	if (_points.empty() || _points[0].empty()) {
		std::cerr << "[Calibrate] Error: No points data available."
				  << std::endl;
		return;
	}

	int m_rows = _points.size();
	int m_cols = _points[0].size();
	int total_views = winSize * winSize;

	// 重新分配内存
	_slice_maps.clear();
	_slice_maps.resize(total_views * 2);

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
			_slice_maps[base_idx] = map_x;
			_slice_maps[base_idx + 1] = map_y;
		}
	}
	// std::cout << "[Calibrate] Slice maps computed." << std::endl;
}

// ---------------------------------------------------------
// 2. 公开接口：强制计算并获取
// ---------------------------------------------------------
void LFCalibrate::computeSliceMaps(int winSize, cv::Mat &out_x, cv::Mat &out_y,
								   int row, int col) {
	_computeSliceMaps(winSize);			  // 强制刷新缓存
	getSliceMaps(out_x, out_y, row, col); // 获取指定视角
}
void LFCalibrate::computeSliceMaps(int winSize) {
	_computeSliceMaps(winSize); // 强制刷新缓存
}

// ---------------------------------------------------------
// 3. 公开接口：获取 (Getter)
// ---------------------------------------------------------
void LFCalibrate::getSliceMaps(cv::Mat &out_x, cv::Mat &out_y, int row,
							   int col) {
	// 安全检查：如果没初始化，直接返回
	if (_slice_maps.empty()) {
		std::cerr << "[Calibrate] Error: Slice maps not initialized. Call "
					 "computeSliceMaps first."
				  << std::endl;
		return;
	}

	// 计算 WinSize (反推)
	int total_maps = _slice_maps.size();
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
	out_x = _slice_maps[idx];
	out_y = _slice_maps[idx + 1];
}

void LFCalibrate::_computeDehexMaps() {
	// 1. 依赖检查：必须先有 Slice Map 才能确定尺寸
	if (_slice_maps.empty()) {
		std::cerr << "[Calibrate] Error: Slice maps are empty. Cannot compute "
					 "Dehex maps."
				  << std::endl;
		return;
	}

	// 2. 从 Slice Map 获取微透镜阵列尺寸
	int grid_rows = _slice_maps[0].rows;
	int grid_cols = _slice_maps[0].cols;

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

	std::cout << "[Calibrate] Dehex maps computed and cached." << std::endl;
}

void LFCalibrate::computeDehexMaps() { _computeDehexMaps(); }
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