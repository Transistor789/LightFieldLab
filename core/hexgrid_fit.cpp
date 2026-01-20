#include "hexgrid_fit.h"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

HexGridFitter::HexGridFitter(const std::pair<cv::Mat, cv::Mat> &pts_mats, bool hex_odd)
	: _x_mat(pts_mats.first), _y_mat(pts_mats.second), _hex_odd(hex_odd) {
	if (_x_mat.empty() || _y_mat.empty()) {
		throw std::invalid_argument("Input coordinate matrices cannot be empty.");
	}
	_rows = _x_mat.rows;
	_cols = _x_mat.cols;
}

void HexGridFitter::fit() {
	if (_rows <= 0 || _cols <= 0)
		throw std::runtime_error("Grid dimensions must be positive");

	const float shift_dir = _hex_odd ? -1.0f : 1.0f;
	std::vector<Eigen::Vector2f> observations;
	std::vector<Eigen::Vector3f> features;

	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 == 0) ? 0.5f * shift_dir : 0.0f;
		for (int c = 0; c < _cols; ++c) {
			// [修改点] 从矩阵获取坐标并传递两个参数给 isValidPoint
			float x = _x_mat.at<float>(r, c);
			float y = _y_mat.at<float>(r, c);

			if (isValidPoint(x, y)) {
				observations.emplace_back(x, y);
				features.emplace_back(r, c + row_shift, 1.0f);
			}
		}
	}

	if (observations.size() < 3)
		throw std::runtime_error("Insufficient points");

	int N = observations.size();
	Eigen::MatrixXf A(N, 3);
	Eigen::MatrixXf B(N, 2);
	for (int i = 0; i < N; ++i) {
		A.row(i) = features[i];
		B.row(i) = observations[i];
	}
	_params = A.colPivHouseholderQr().solve(B);
	Eigen::MatrixXf residuals = A * _params - B;
	float mse = residuals.squaredNorm() / N;
	_rmse = std::sqrt(mse);
	_is_fitted = true;
}

// =========================================================
// 新增 RANSAC 鲁棒拟合实现
// =========================================================
void HexGridFitter::fitRobust(float threshold) {
	if (_rows <= 0 || _cols <= 0) {
		throw std::runtime_error("Grid dimensions must be positive");
	}

	const float shift_dir = _hex_odd ? 1.0f : -1.0f;

	struct DataPoint {
		Eigen::Vector3f feat;
		Eigen::Vector2f obs;
	};
	std::vector<DataPoint> data;
	data.reserve(_rows * _cols); // [修改点] 使用矩阵尺寸预分配

	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
		for (int c = 0; c < _cols; ++c) {
			// [修改点] 统一使用矩阵访问
			float x = _x_mat.at<float>(r, c);
			float y = _y_mat.at<float>(r, c);

			if (isValidPoint(x, y)) {
				data.push_back({Eigen::Vector3f(r, c + row_shift, 1.0f), Eigen::Vector2f(x, y)});
			}
		}
	}
	// ... 剩下的 RANSAC 逻辑保持之前提供的矩阵版本即可 ...
}

void HexGridFitter::fitFastRobust(float threshold, int sample_size) {
	if (_rows <= 0 || _cols <= 0)
		throw std::runtime_error("Invalid grid dimensions");

	// 1. 数据准备：从矩阵中收集有效点
	struct DataPoint {
		float r_idx, c_idx, x, y;
	};
	std::vector<DataPoint> all_data;
	all_data.reserve(_rows * _cols);

	const float shift_dir = _hex_odd ? 1.0f : -1.0f;
	for (int r = 0; r < _rows; ++r) {
		bool is_physically_shifted = _hex_odd ? (r % 2 == 0) : (r % 2 != 0);
		float row_shift = is_physically_shifted ? 0.5f : 0.0f;

		for (int c = 0; c < _cols; ++c) {
			float x = _x_mat.at<float>(r, c);
			float y = _y_mat.at<float>(r, c);
			if (isValidPoint(x, y)) {
				all_data.push_back({(float)r, (float)c + row_shift, x, y});
			}
		}
	}

	int N = static_cast<int>(all_data.size());
	if (N < 3)
		throw std::runtime_error("Insufficient valid points for fitting");

	// 2. RANSAC 核心逻辑
	std::vector<DataPoint> subset = all_data;
	std::mt19937 g(42);
	if (N > sample_size) {
		std::shuffle(subset.begin(), subset.end(), g);
		subset.resize(sample_size);
	}

	int best_inliers = -1;
	Eigen::Matrix<float, 3, 2> best_model = Eigen::MatrixXf::Zero(3, 2);

	for (int iter = 0; iter < 100; ++iter) {
		// 随机选 3 个点
		std::vector<int> idx(3);
		std::uniform_int_distribution<int> dist(0, subset.size() - 1);
		for (int i = 0; i < 3; ++i) idx[i] = dist(g);

		Eigen::Matrix3f A_local;
		Eigen::Matrix<float, 3, 2> B_local;
		for (int i = 0; i < 3; ++i) {
			A_local.row(i) << subset[idx[i]].r_idx, subset[idx[i]].c_idx, 1.0f;
			B_local.row(i) << subset[idx[i]].x, subset[idx[i]].y;
		}

		// 求解局部模型
		Eigen::Matrix<float, 3, 2> model = A_local.colPivHouseholderQr().solve(B_local);

		// 统计内点
		int inliers = 0;
		for (const auto &p : subset) {
			Eigen::Vector3f feat(p.r_idx, p.c_idx, 1.0f);
			Eigen::Vector2f pred = model.transpose() * feat;
			float dist_sq = std::pow(pred(0) - p.x, 2) + std::pow(pred(1) - p.y, 2);
			if (dist_sq < threshold * threshold)
				inliers++;
		}

		if (inliers > best_inliers) {
			best_inliers = inliers;
			best_model = model;
		}
	}

	// 3. 最终精炼：使用所有内点进行最小二乘拟合
	std::vector<DataPoint> inlier_points;
	for (const auto &p : all_data) {
		Eigen::Vector3f feat(p.r_idx, p.c_idx, 1.0f);
		Eigen::Vector2f pred = best_model.transpose() * feat;
		float dist_sq = std::pow(pred(0) - p.x, 2) + std::pow(pred(1) - p.y, 2);
		if (dist_sq < threshold * threshold)
			inlier_points.push_back(p);
	}

	int M = inlier_points.size();
	Eigen::MatrixXf A_final(M, 3);
	Eigen::MatrixXf B_final(M, 2);
	for (int i = 0; i < M; ++i) {
		A_final.row(i) << inlier_points[i].r_idx, inlier_points[i].c_idx, 1.0f;
		B_final.row(i) << inlier_points[i].x, inlier_points[i].y;
	}
	_params = A_final.colPivHouseholderQr().solve(B_final);

	// 4. 计算 RMSE
	Eigen::MatrixXf res = A_final * _params - B_final;
	_rmse = std::sqrt(res.squaredNorm() / M);
	_is_fitted = true;
}

// =========================================================

std::pair<cv::Mat, cv::Mat> HexGridFitter::predict() const {
	if (!_is_fitted)
		throw std::runtime_error("Must call fit() before predict()");

	cv::Mat pred_x(_rows, _cols, CV_32F);
	cv::Mat pred_y(_rows, _cols, CV_32F);

	const float shift_dir = _hex_odd ? 1.0f : -1.0f;

	for (int r = 0; r < _rows; ++r) {
		bool is_shifted_row = _hex_odd ? (r % 2 == 0) : (r % 2 != 0);
		const float row_shift = is_shifted_row ? 0.5f * shift_dir : 0.0f;
		for (int c = 0; c < _cols; ++c) {
			Eigen::Vector3f feat(r, c + row_shift, 1.0f);
			Eigen::Matrix<float, 2, 1> pred = _params.transpose() * feat;
			pred_x.at<float>(r, c) = pred(0);
			pred_y.at<float>(r, c) = pred(1);
		}
	}

	GridInfo info = get_grid_info(); // 计算拟合后的 Pitch 和 RMSE
	std::cout << "[HexGridFitter::predict] Fitting Results:" << std::endl;
	std::cout << std::format("  > Grid Size: [{}, {}]", _cols, _rows) << std::endl;
	std::cout << std::format("  > Pitch:     [{:.4f}, {:.4f}]", info.pitch_col, info.pitch_row) << std::endl;
	std::cout << std::format("  > RMSE:      {:.6f}", info.rmse) << std::endl;

	return {pred_x, pred_y};
}

HexGridFitter::GridInfo HexGridFitter::get_grid_info() const {
	if (!_is_fitted) {
		throw std::runtime_error("Must call fit() before get_grid_info()");
	}

	cv::Point2f origin(_params(2, 0), _params(2, 1)); // [a2, b2]
	cv::Point2f vec_v(_params(0, 0), _params(0, 1));  // [a0, b0]
	cv::Point2f vec_h(_params(1, 0), _params(1, 1));  // [a1, b1]

	float pitch_v = std::sqrt(vec_v.x * vec_v.x + vec_v.y * vec_v.y);
	float pitch_h = std::sqrt(vec_h.x * vec_h.x + vec_h.y * vec_h.y);

	return GridInfo{origin, vec_v, vec_h, pitch_v, pitch_h, _rmse};
}

// 仅保留这一个实现
bool HexGridFitter::isValidPoint(float x, float y) const {
	return x >= 0 && y >= 0 && !std::isnan(x) && !std::isnan(y);
}