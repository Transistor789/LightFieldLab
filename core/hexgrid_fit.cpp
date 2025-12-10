#include "hexgrid_fit.h"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>

HexGridFitter::HexGridFitter(const std::vector<cv::Point2f> &pts_sorted,
							 const std::vector<int> &pts_size, bool hex_odd)
	: _pts_sorted(pts_sorted), _hex_odd(hex_odd) {
	if (pts_size.size() != 2) {
		throw std::invalid_argument(
			"pts_size must have 2 elements: {cols, rows}");
	}
	_cols = pts_size[0];
	_rows = pts_size[1];
	if (static_cast<int>(_pts_sorted.size()) != _rows * _cols) {
		throw std::invalid_argument(
			"pts_sorted size mismatch with rows * cols");
	}
}

void HexGridFitter::fit() {
	if (_rows <= 0 || _cols <= 0) {
		throw std::runtime_error("Grid dimensions must be positive");
	}

	const float shift_dir = _hex_odd ? 1.0f : -1.0f;

	// 准备观测值和特征矩阵
	std::vector<Eigen::Vector2f> observations;
	std::vector<Eigen::Vector3f> features;

	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 != 0) ? 0.5f * shift_dir : 0.0f;

		for (int c = 0; c < _cols; ++c) {
			const cv::Point2f &pt = _pts_sorted[r * _cols + c];
			if (isValidPoint(pt)) {
				observations.emplace_back(pt.x, pt.y);		   // y, x
				features.emplace_back(r, c + row_shift, 1.0f); // v, u_eff, 1
			}
		}
	}

	if (observations.size() < 3) {
		throw std::runtime_error("Insufficient valid points for fitting");
	}

	// 构建矩阵 A (N x 3), B (N x 2)
	int N = static_cast<int>(observations.size());
	Eigen::MatrixXf A(N, 3);
	Eigen::MatrixXf B(N, 2);

	for (int i = 0; i < N; ++i) {
		A.row(i) = features[i];
		B.row(i) = observations[i];
	}

	// 求解 AX = B → X = A^+ B (最小二乘解)
	_params = A.colPivHouseholderQr().solve(B); // 3x2

	// 计算 RMSE
	Eigen::MatrixXf residuals = A * _params - B;
	float mse = residuals.squaredNorm() / N;
	_rmse = std::sqrt(mse);

	_is_fitted = true;
}

std::vector<std::vector<cv::Point2f>> HexGridFitter::predict() const {
	if (!_is_fitted) {
		throw std::runtime_error("Must call fit() before predict()");
	}

	std::vector<std::vector<cv::Point2f>> grid(_rows,
											   std::vector<cv::Point2f>(_cols));

	const float shift_dir = _hex_odd ? 1.0f : -1.0f;

	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 != 0) ? 0.5f * shift_dir : 0.0f;

		for (int c = 0; c < _cols; ++c) {
			Eigen::Vector3f feat(r, c + row_shift, 1.0f);
			// 明确指定矩阵乘法
			Eigen::Matrix<float, 2, 1> pred = _params.transpose() * feat;
			grid[r][c] = cv::Point2f(pred(0), pred(1)); // (y, x)
		}
	}

	return grid;
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

bool HexGridFitter::isValidPoint(const cv::Point2f &pt) const {
	return !std::isnan(pt.x) && !std::isnan(pt.y);
}