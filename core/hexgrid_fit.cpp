#include "hexgrid_fit.h"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// ... (保持原有的构造函数和 fit() 不变) ...

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
	// ... (保持原有代码不变)
	// 为了节省篇幅，这里省略，请保留您文件中的原样
	if (_rows <= 0 || _cols <= 0)
		throw std::runtime_error("Grid dimensions must be positive");
	const float shift_dir = _hex_odd ? 1.0f : -1.0f;
	std::vector<Eigen::Vector2f> observations;
	std::vector<Eigen::Vector3f> features;
	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
		for (int c = 0; c < _cols; ++c) {
			const cv::Point2f &pt = _pts_sorted[r * _cols + c];
			if (isValidPoint(pt)) {
				observations.emplace_back(pt.x, pt.y);
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

	// 1. 收集有效点 (过滤掉 -1 和 NaN)
	struct DataPoint {
		Eigen::Vector3f feat; // r, u_eff, 1
		Eigen::Vector2f obs;  // y, x
	};
	std::vector<DataPoint> data;
	data.reserve(_pts_sorted.size());

	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
		for (int c = 0; c < _cols; ++c) {
			const cv::Point2f &pt = _pts_sorted[r * _cols + c];

			// [关键修复] 显式检查负坐标，剔除 (-1, -1)
			if (pt.x >= 0 && pt.y >= 0 && isValidPoint(pt)) {
				data.push_back({Eigen::Vector3f(r, c + row_shift, 1.0f),
								Eigen::Vector2f(pt.x, pt.y)});
			}
		}
	}

	int N = static_cast<int>(data.size());
	if (N < 3) {
		throw std::runtime_error("Insufficient valid points for RANSAC");
	}

	// 2. RANSAC 参数
	int max_iters = 200; // 迭代次数
	int best_inliers = 0;
	Eigen::Matrix<float, 3, 2> best_model;
	best_model.setZero();

	std::mt19937 rng(42);
	std::uniform_int_distribution<int> dist(0, N - 1);

	// 3. RANSAC 循环
	for (int iter = 0; iter < max_iters; ++iter) {
		// a. 随机采样 3 点
		int idx1 = dist(rng);
		int idx2 = dist(rng);
		int idx3 = dist(rng);

		// 简单去重
		if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
			continue;

		const auto &p1 = data[idx1];
		const auto &p2 = data[idx2];
		const auto &p3 = data[idx3];

		// b. 求解仿射模型 (3点解)
		// Model: Obs^T = A * Feat^T  => A = Obs^T * Feat^T_inv
		// Feat Matrix (3x3): [f1, f2, f3]
		// Obs Matrix (2x3):  [o1, o2, o3]
		Eigen::Matrix3f F;
		F << p1.feat, p2.feat, p3.feat; // columns

		// 检查共线 (行列式接近0)
		if (std::abs(F.determinant()) < 1e-4)
			continue;

		Eigen::Matrix<float, 2, 3> O;
		O << p1.obs, p2.obs, p3.obs; // columns

		// A_trans = (O * F_inv).transpose()
		Eigen::Matrix<float, 3, 2> model = (O * F.inverse()).transpose();

		// c. 统计内点
		int inliers = 0;
		float thresh_sq = threshold * threshold;
		for (int i = 0; i < N; ++i) {
			// Pred = model^T * feat
			Eigen::Vector2f pred = model.transpose() * data[i].feat;
			float err_sq = (pred - data[i].obs).squaredNorm();
			if (err_sq < thresh_sq) {
				inliers++;
			}
		}

		// d. 更新最佳模型
		if (inliers > best_inliers) {
			best_inliers = inliers;
			best_model = model;
		}
	}

	// 4. 精炼 (Refine)：使用所有内点做最小二乘
	std::vector<Eigen::Vector3f> inlier_feats;
	std::vector<Eigen::Vector2f> inlier_obs;
	inlier_feats.reserve(N);
	inlier_obs.reserve(N);

	float thresh_sq = threshold * threshold;
	for (int i = 0; i < N; ++i) {
		Eigen::Vector2f pred = best_model.transpose() * data[i].feat;
		if ((pred - data[i].obs).squaredNorm() < thresh_sq) {
			inlier_feats.push_back(data[i].feat);
			inlier_obs.push_back(data[i].obs);
		}
	}

	// 如果内点太少，就退化回原始模型
	if (inlier_feats.size() < 3) {
		_params = best_model;
	} else {
		int M = inlier_feats.size();
		Eigen::MatrixXf A_final(M, 3);
		Eigen::MatrixXf B_final(M, 2);
		for (int i = 0; i < M; ++i) {
			A_final.row(i) = inlier_feats[i];
			B_final.row(i) = inlier_obs[i];
		}
		_params = A_final.colPivHouseholderQr().solve(B_final);
	}

	// 5. 计算最终 RMSE (只算内点，或算全部有效点)
	// 这里计算相对于所有有效点的 RMSE，以反映真实覆盖情况
	float total_err_sq = 0.0f;
	int valid_count = 0;
	for (int i = 0; i < N; ++i) {
		// 只计算距离不太离谱的点的误差，避免(-1,-1)干扰显示
		Eigen::Vector2f pred = _params.transpose() * data[i].feat;
		float err_sq = (pred - data[i].obs).squaredNorm();
		if (err_sq < 1000.0f) { // 过滤掉极其离谱的噪点影响统计
			total_err_sq += err_sq;
			valid_count++;
		}
	}
	_rmse = (valid_count > 0) ? std::sqrt(total_err_sq / valid_count) : 0.0f;
	_is_fitted = true;
}

void HexGridFitter::fitFastRobust(float threshold, int sample_size) {
	if (_rows <= 0 || _cols <= 0) {
		throw std::runtime_error("Grid dimensions must be positive");
	}

	// 1. 数据准备：收集所有有效点
	// Data: {r, c, y, x}
	struct DataPoint {
		int r, c;
		float x, y; // <--- 改为 x, y
	};
	std::vector<DataPoint> all_data;
	all_data.reserve(_pts_sorted.size());

	for (int r = 0; r < _rows; ++r) {
		for (int c = 0; c < _cols; ++c) {
			const cv::Point2f &pt = _pts_sorted[r * _cols + c];
			// 过滤 (-1,-1) 和 NaN
			if (pt.x > 0 && pt.y > 0 && isValidPoint(pt)) {
				all_data.push_back(
					{r, c, pt.x, pt.y}); // 注意 Eigen 存的是 (y, x)
			}
		}
	}

	int N = static_cast<int>(all_data.size());
	if (N < 3)
		throw std::runtime_error("Insufficient valid points");

	// 2. 创建随机子集 (Subset) 用于加速 RANSAC
	std::vector<DataPoint> subset;
	if (N > sample_size) {
		subset = all_data; // Copy
		std::mt19937 g(42);
		std::shuffle(subset.begin(), subset.end(), g);
		subset.resize(sample_size);
	} else {
		subset = all_data;
	}
	int M = subset.size();

	// 3. RANSAC 循环
	int max_iters = 500; // Debug 模式下 500 次也非常快 (因为只算 M 个点)
	int best_inliers = -1;
	Eigen::Matrix<float, 3, 2> best_model;
	bool best_hex_odd = _hex_odd; // 记录最佳的奇偶性

	std::mt19937 rng(123);
	std::uniform_int_distribution<int> dist(0, M - 1);
	float thresh_sq = threshold * threshold;

	for (int iter = 0; iter < max_iters; ++iter) {
		// a. 采样 3 点
		int idx1 = dist(rng);
		int idx2 = dist(rng);
		int idx3 = dist(rng);
		if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
			continue;

		const auto &p1 = subset[idx1];
		const auto &p2 = subset[idx2];
		const auto &p3 = subset[idx3];

		// b. 尝试两种奇偶性 (True/False)
		// 因为如果 Parity 错了，模型会完全错误，所以我们让 RANSAC 自己去试
		bool parities[2] = {false, true};

		for (bool try_odd : parities) {
			float shift_dir = try_odd ? 1.0f : -1.0f;

			// 构建特征矩阵 F
			Eigen::Matrix3f F;
			auto get_feat = [&](const DataPoint &p) -> Eigen::Vector3f {
				float row_shift = (p.r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
				return Eigen::Vector3f(p.r, p.c + row_shift, 1.0f);
			};

			F << get_feat(p1), get_feat(p2), get_feat(p3);

			if (std::abs(F.determinant()) < 1e-4)
				continue; // 共线

			// 构建观测矩阵 O (2x3) -> Transposed in memory logic
			// Obs: [y, x]
			Eigen::Matrix<float, 2, 3> O;
			O << p1.x, p2.x, p3.x, // Row 0: x (横坐标)
				p1.y, p2.y, p3.y;  // Row 1: y (纵坐标)

			// Model = (O * F_inv)^T
			Eigen::Matrix<float, 3, 2> model = (O * F.inverse()).transpose();

			// c. 在子集上验证
			int inliers = 0;
			for (int i = 0; i < M; ++i) {
				Eigen::Vector2f obs(subset[i].x, subset[i].y);
				Eigen::Vector2f pred = model.transpose() * get_feat(subset[i]);
				if ((pred - obs).squaredNorm() < thresh_sq) {
					inliers++;
				}
			}

			// d. 更新最佳
			if (inliers > best_inliers) {
				best_inliers = inliers;
				best_model = model;
				best_hex_odd = try_odd;
			}
		}
	}

	// 4. 更新类的奇偶性设置 (关键修正：解决 RMSE=10)
	_hex_odd = best_hex_odd;
	float shift_dir = _hex_odd ? 1.0f : -1.0f;

	// 5. Global Refine (在全量数据上精炼)
	// 使用最佳模型过滤全量数据，收集所有内点
	std::vector<Eigen::Vector3f> final_feats;
	std::vector<Eigen::Vector2f> final_obs;
	final_feats.reserve(N);
	final_obs.reserve(N);

	// 此时阈值可以稍宽一点，或者保持
	for (const auto &d : all_data) {
		float row_shift = (d.r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
		Eigen::Vector3f feat(d.r, d.c + row_shift, 1.0f);
		Eigen::Vector2f obs(d.x, d.y);

		Eigen::Vector2f pred = best_model.transpose() * feat;
		if ((pred - obs).squaredNorm() < thresh_sq) {
			final_feats.push_back(feat);
			final_obs.push_back(obs);
		}
	}

	// 最小二乘求解
	if (final_feats.size() < 3) {
		_params = best_model; // 退化回 RANSAC 结果
	} else {
		int K = final_feats.size();
		Eigen::MatrixXf A(K, 3);
		Eigen::MatrixXf B(K, 2);
		for (int i = 0; i < K; ++i) {
			A.row(i) = final_feats[i];
			B.row(i) = final_obs[i];
		}
		_params = A.colPivHouseholderQr().solve(B);
	}

	// 6. 计算最终 RMSE
	float total_err = 0.0f;
	int count = 0;
	// 统计所有"靠谱"点的误差
	for (const auto &d : all_data) {
		float row_shift = (d.r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
		Eigen::Vector3f feat(d.r, d.c + row_shift, 1.0f);
		Eigen::Vector2f obs(d.x, d.y);
		Eigen::Vector2f pred = _params.transpose() * feat;
		float err = (pred - obs).squaredNorm();

		// 仅统计内点或接近内点的误差，避免离群点拉高 RMSE
		if (err < 100.0f) {
			total_err += err;
			count++;
		}
	}
	_rmse = count > 0 ? std::sqrt(total_err / count) : 0.0f;
	_is_fitted = true;
}

// =========================================================

std::vector<std::vector<cv::Point2f>> HexGridFitter::predict() const {
	// ... (保持不变) ...
	if (!_is_fitted)
		throw std::runtime_error("Must call fit() before predict()");
	std::vector<std::vector<cv::Point2f>> grid(_rows,
											   std::vector<cv::Point2f>(_cols));
	const float shift_dir = _hex_odd ? 1.0f : -1.0f;
	for (int r = 0; r < _rows; ++r) {
		const float row_shift = (r % 2 != 0) ? 0.5f * shift_dir : 0.0f;
		for (int c = 0; c < _cols; ++c) {
			Eigen::Vector3f feat(r, c + row_shift, 1.0f);
			Eigen::Matrix<float, 2, 1> pred = _params.transpose() * feat;
			grid[r][c] = cv::Point2f(pred(0), pred(1));
		}
	}
	GridInfo info = get_grid_info();
	std::cout << "[HexGridFitter] Results:" << std::endl;
	std::cout << "  > Grid Size: " << _cols << " x " << _rows << std::endl;
	std::cout << "  > Pitch: [" << info.pitch_col << ", " << info.pitch_row
			  << "]" << std::endl;
	std::cout << "  > RMSE:      " << info.rmse << std::endl;
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