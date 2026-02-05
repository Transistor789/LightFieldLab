#include "lfcalibrate.h"

#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "json.hpp"
#include "utils.h"

#include <format>
#include <iostream>
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
		temp = img.clone();
	}

	// 1. 预处理：消除 Bayer 棋盘格
	if (config.bayer != BayerPattern::NONE) {
		// 使用高斯模糊平滑 Bayer 纹理，保留几何质心
		cv::GaussianBlur(temp, temp, cv::Size(9, 9), 0);
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
	cs.run2(config.rot); // 使用泛洪填充算法
	_maps = cs.getPointsAsMats(config.crop);
	cv::Mat x_mat = _maps.first;
	_hex_odd = x_mat.at<float>(0, 0) < x_mat.at<float>(0, 1);
	// _hex_odd = cs.getHexOdd();

	if (config.hexgridfit) {
		// 5. 网格拟合 (HexGrid Fit)
		_fitter = std::make_shared<HexGridFitter>(_maps, _hex_odd);

		// 使用快速鲁棒拟合
		_fitter->fitFastRobust(2.0f, 1500);
		_maps = _fitter->predict();
	}
	if (config.orientation == Orientation::VERT) {
		cv::transpose(_maps.first, _maps.first);
		cv::transpose(_maps.second, _maps.second);
		std::swap(_maps.first, _maps.second);
	}
	if (config.genLUT) {
		// computeExtractMaps(config.views, config.space);
		// computeExtractMaps(config.views, config.space, config.rot);
		computeExtractMapsStepByStep(config.views, config.space, config.rot);
		computeDehexMaps();
		if (config.orientation == Orientation::VERT) {
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

const std::vector<cv::Mat> &LFCalibrate::computeExtractMaps(int winSize, float space, float rad) {
	rad = 0.0f;
	if (_maps.first.empty() || _maps.second.empty()) {
		std::cerr << "[LFCalibrate] Error: No maps data." << std::endl;
		return _extract_maps;
	}

	// 1. 计算旋转分量
	float cos_r = std::cos(rad);
	float sin_r = std::sin(rad);

	int m_rows = _maps.first.rows;
	int m_cols = _maps.first.cols;
	int total_views = winSize * winSize;

	_extract_maps.clear();
	_extract_maps.resize(total_views * 2);

	float startOffset = -(winSize - 1) / 2.0f;

#pragma omp parallel for
	for (int u = 0; u < winSize; ++u) {
		for (int v = 0; v < winSize; ++v) {
			// 2. 原始未旋转的逻辑偏移量
			float raw_off_y = (startOffset + u) * space;
			float raw_off_x = (startOffset + v) * space;

			// 3. 应用旋转矩阵进行坐标变换
			// 注意：这里旋转的是相对于中心点的“偏移向量”
			float rot_off_x = raw_off_x * cos_r - raw_off_y * sin_r;
			float rot_off_y = raw_off_x * sin_r + raw_off_y * cos_r;

			// 4. 将旋转后的偏移量应用到全局中心点矩阵上
			cv::Mat map_x = _maps.first + rot_off_x;
			cv::Mat map_y = _maps.second + rot_off_y;

			int idx = (u * winSize + v) * 2;
			_extract_maps[idx] = map_x;
			_extract_maps[idx + 1] = map_y;
		}
	}

	std::cout << std::format("[LFCalibrate] Extract maps computed with Rotation: {:.4f} deg, Space: {:.2f}", rad, space)
			  << std::endl;

	return _extract_maps;
}

const std::vector<cv::Mat> &LFCalibrate::computeExtractMapsStepByStep(int winSize, float space, float rad) {
	rad *= -0.0;
	if (_maps.first.empty() || _maps.second.empty()) {
		std::cerr << "[LFCalibrate] Error: No maps data." << std::endl;
		return _extract_maps;
	}

	int m_rows = _maps.first.rows; // 434
	int m_cols = _maps.first.cols; // 625
	int total_views = winSize * winSize;
	float startOffset = -(winSize - 1) / 2.0f;

	// 扁平化连续内存，存储所有采样点
	// 布局索引：((r * m_cols + c) * total_views) + (u * winSize + v)
	std::vector<cv::Point2f> flat_points(m_rows * m_cols * total_views);

// --- 第一步：生成初始采样点 (未旋转) ---
#pragma omp parallel for
	for (int r = 0; r < m_rows; ++r) {
		for (int c = 0; c < m_cols; ++c) {
			float cx = _maps.first.at<float>(r, c);
			float cy = _maps.second.at<float>(r, c);
			int lens_offset = (r * m_cols + c) * total_views;

			for (int u = 0; u < winSize; ++u) {
				for (int v = 0; v < winSize; ++v) {
					int pt_idx = lens_offset + (u * winSize + v);
					// 存储绝对坐标（未旋转）
					flat_points[pt_idx].x = cx + (startOffset + v) * space;
					flat_points[pt_idx].y = cy + (startOffset + u) * space;
					// if (r == m_rows / 2 && c == m_cols / 2) {
					// 	std::cout << std::format("[Step 1 - Original] Lens({},{}) View({},{}): ({:.4f}, {:.4f})\n", r,
					// 							 c, u, v, flat_points[pt_idx].x, flat_points[pt_idx].y);
					// }
				}
			}
		}
	}

	// --- 调试：提取并输出中心透镜的 9x9 采样矩阵 ---
	// int r_mid = 0;
	// int c_mid = 0;
	// // int r_mid = m_rows / 2;
	// // int c_mid = m_cols / 2;
	// int debug_lens_offset = (r_mid * m_cols + c_mid) * total_views;

	// cv::Mat debug_x(winSize, winSize, CV_32F);
	// cv::Mat debug_y(winSize, winSize, CV_32F);

	// for (int u = 0; u < winSize; ++u) {
	// 	for (int v = 0; v < winSize; ++v) {
	// 		int pt_idx = debug_lens_offset + (u * winSize + v);
	// 		debug_x.at<float>(u, v) = flat_points[pt_idx].x;
	// 		debug_y.at<float>(u, v) = flat_points[pt_idx].y;
	// 	}
	// }

	// // 直接使用 OpenCV 的格式化输出矩阵信息
	// std::cout << std::format("\n[Debug] Center Lens({},{}) 9x9 X-Sampling Matrix:\n", r_mid, c_mid) << debug_x
	// 		  << std::endl;
	// std::cout << std::format("\n[Debug] Center Lens({},{}) 9x9 Y-Sampling Matrix:\n", r_mid, c_mid) << debug_y
	// 		  << std::endl;

	// --- 第二步：应用旋转变换 ---
	float cos_r = std::cos(rad);
	float sin_r = std::sin(rad);

#pragma omp parallel for
	for (int r = 0; r < m_rows; ++r) {
		for (int c = 0; c < m_cols; ++c) {
			float cx = _maps.first.at<float>(r, c);
			float cy = _maps.second.at<float>(r, c);
			int lens_offset = (r * m_cols + c) * total_views;

			for (int uv = 0; uv < total_views; ++uv) {
				int pt_idx = lens_offset + uv;

				// 1. 计算相对于中心点的偏移
				float dx = flat_points[pt_idx].x - cx;
				float dy = flat_points[pt_idx].y - cy;

				// 2. 旋转偏移量并重新写回绝对坐标
				flat_points[pt_idx].x = cx + (dx * cos_r - dy * sin_r);
				flat_points[pt_idx].y = cy + (dx * sin_r + dy * cos_r);
				// if (r == m_rows / 2 && c == m_cols / 2) {
				// 	// 还原 u, v 索引用于打印展示
				// 	int u_idx = uv / winSize;
				// 	int v_idx = uv % winSize;
				// 	std::cout << std::format(
				// 		"[Step 2 - Rotated]  Lens({},{}) View({},{}): ({:.4f}, {:.4f}) | Offset: ({:.4f}, {:.4f})\n", r,
				// 		c, u_idx, v_idx, flat_points[pt_idx].x, flat_points[pt_idx].y, flat_points[pt_idx].x - cx,
				// 		flat_points[pt_idx].y - cy);
				// }
			}
		}
	}

	// for (int u = 0; u < winSize; ++u) {
	// 	for (int v = 0; v < winSize; ++v) {
	// 		int pt_idx = debug_lens_offset + (u * winSize + v);
	// 		debug_x.at<float>(u, v) = flat_points[pt_idx].x;
	// 		debug_y.at<float>(u, v) = flat_points[pt_idx].y;
	// 	}
	// }
	// std::cout << std::format("\n[Debug] Center Lens({},{}) 9x9 X-Sampling Matrix:\n", r_mid, c_mid) << debug_x
	// 		  << std::endl;
	// std::cout << std::format("\n[Debug] Center Lens({},{}) 9x9 Y-Sampling Matrix:\n", r_mid, c_mid) << debug_y
	// 		  << std::endl;
	// --- [此处可插入代码：打印旋转后的特定点坐标进行对比] ---

	// --- 第三步：重排为视角地图 (cv::Mat) ---
	_extract_maps.clear();
	_extract_maps.resize(total_views * 2);

	for (int uv = 0; uv < total_views; ++uv) {
		int u = uv / winSize;
		int v = uv % winSize;

		cv::Mat map_x(m_rows, m_cols, CV_32F);
		cv::Mat map_y(m_rows, m_cols, CV_32F);

#pragma omp parallel for
		for (int r = 0; r < m_rows; ++r) {
			float *ptr_x = map_x.ptr<float>(r);
			float *ptr_y = map_y.ptr<float>(r);
			for (int c = 0; c < m_cols; ++c) {
				int pt_idx = (r * m_cols + c) * total_views + uv;
				ptr_x[c] = flat_points[pt_idx].x;
				ptr_y[c] = flat_points[pt_idx].y;
			}
		}

		int idx = uv * 2;
		_extract_maps[idx] = map_x;
		_extract_maps[idx + 1] = map_y;
	}

	std::cout << std::format("[LFCalibrate] Extract maps computed with Rotation: {:.4f} deg, Space: {:.2f}", rad, space)
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
