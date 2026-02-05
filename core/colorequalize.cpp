#include "colorequalize.h"

#include <cmath>
#include <omp.h>
#include <opencv2/core/cuda.hpp> // GpuMat 基础
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp> // meanStdDev, cvtColor 等
#include <vector>

// =========================================================
// Public Interface
// =========================================================

void ColorEqualize::equalize(std::vector<cv::Mat> &views, ColorEqualizeMethod method) {
	if (views.empty())
		return;

	// --- 增加全局检查：如果第一张图不是 8-bit RGB，直接退出整个流程 ---
	if (views[0].type() != CV_8UC3) {
		return;
	}

	int num_views = static_cast<int>(views.size());
	int side_len = static_cast<int>(std::sqrt(num_views));
	int center_idx = (side_len / 2) * side_len + (side_len / 2);
	if (center_idx >= num_views)
		center_idx = 0;

	const cv::Mat &ref_img = views[center_idx];

	if (method == ColorEqualizeMethod::Reinhard) {
		cv::Scalar ref_mean, ref_std;
		computeLabStats(ref_img, ref_mean, ref_std);

		// 如果参考图统计失败（虽然理论上不会），则退出
		if (ref_mean == cv::Scalar::all(0) && ref_std == cv::Scalar::all(0))
			return;

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_views; ++i) {
			if (i == center_idx)
				continue;
			reinhard(views[i], ref_mean, ref_std);
		}
	} else {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_views; ++i) {
			if (i == center_idx)
				continue;
			apply(views[i], ref_img, method);
		}
	}
}

void ColorEqualize::apply(cv::Mat &src, const cv::Mat &ref, ColorEqualizeMethod method) {
	if (src.empty() || ref.empty())
		return;

	switch (method) {
		case ColorEqualizeMethod::Reinhard:
			break;
		case ColorEqualizeMethod::HistMatch:
			histMatch(src, ref);
			break;
		case ColorEqualizeMethod::MKL:
			mkl(src, ref);
			break;
		case ColorEqualizeMethod::MVGD:
			mvgd(src, ref);
			break;
		case ColorEqualizeMethod::HM_MKL_HM:
			histMatch(src, ref);
			mkl(src, ref);
			histMatch(src, ref);
			break;
		case ColorEqualizeMethod::HM_MVGD_HM:
			histMatch(src, ref);
			mvgd(src, ref);
			histMatch(src, ref);
			break;
	}
}

// =========================================================
// MKL & MVGD Implementation (Linear Color Transfer)
// =========================================================

// 辅助：计算均值和协方差矩阵 (Result: mean 1x3, cov 3x3, CV_64F)
void ColorEqualize::computeMeanCov(const cv::Mat &img, cv::Mat &mean, cv::Mat &cov) {
	// 依然建议在这里保留转换，因为协方差累加容易溢出 float 精度
	cv::Mat float_img;
	img.reshape(1, img.total()).convertTo(float_img, CV_64F);
	cv::calcCovarMatrix(float_img, cov, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
}

// 辅助：计算矩阵平方根 A^(1/2)
cv::Mat ColorEqualize::sqrtMatrix(const cv::Mat &cov) {
	int n = cov.rows; // 动态获取通道数
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat sqrt_lambda = cv::Mat::zeros(n, n, CV_64F);
	for (int i = 0; i < n; ++i) {
		double val = eigenvalues.at<double>(i);
		sqrt_lambda.at<double>(i, i) = (val > 0) ? std::sqrt(val) : 0.0;
	}
	return eigenvectors.t() * sqrt_lambda * eigenvectors;
}

// 辅助：计算矩阵逆平方根 A^(-1/2)
cv::Mat ColorEqualize::invSqrtMatrix(const cv::Mat &cov) {
	int n = cov.rows; // 动态获取通道数
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat inv_sqrt_lambda = cv::Mat::zeros(n, n, CV_64F);
	for (int i = 0; i < n; ++i) {
		double val = eigenvalues.at<double>(i);
		inv_sqrt_lambda.at<double>(i, i) = (val > 1e-9) ? (1.0 / std::sqrt(val)) : 0.0;
	}
	return eigenvectors.t() * inv_sqrt_lambda * eigenvectors;
}

void ColorEqualize::mkl(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty() || src.type() != CV_8UC3 || ref.type() != CV_8UC3)
		return;

	// 1. 计算统计量 (使用 CV_64F 保证矩阵分解精度)
	cv::Mat mu_s, cov_s, mu_r, cov_r;
	computeMeanCov(src, mu_s, cov_s);
	computeMeanCov(ref, mu_r, cov_r);

	// 2. 计算 MKL 变换矩阵 T
	cv::Mat cov_s_sqrt = sqrtMatrix(cov_s);
	cv::Mat cov_s_inv_sqrt = invSqrtMatrix(cov_s);
	cv::Mat C = cov_s_sqrt * cov_r * cov_s_sqrt;
	cv::Mat C_sqrt = sqrtMatrix(C);
	cv::Mat T = cov_s_inv_sqrt * C_sqrt * cov_s_inv_sqrt;

	// 3. 性能优化：将 P_out = (P_in - mu_s) * T + mu_r 转换为 P_out = P_in * T + B
	// 其中 B = mu_r - mu_s * T
	cv::Mat B = mu_r - mu_s * T;

	// 4. 调用 cv::transform 执行高效的线性映射 (取代 reshape + repeat)
	// 注意：cv::transform 接受的是 T 的转置形式
	cv::Mat T_float, B_float, M;
	T.convertTo(T_float, CV_32F);
	B.convertTo(B_float, CV_32F);

	// 构造 cv::transform 所需的 3x4 变换矩阵 [T | B]
	cv::hconcat(T_float.t(), B_float.t(), M);

	cv::transform(src, src, M);
}

void ColorEqualize::mvgd(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty() || src.type() != CV_8UC3 || ref.type() != CV_8UC3)
		return;

	if (src.size() != ref.size()) {
		mkl(src, ref);
		return;
	}

	// 1. 数据准备 (只需一次转换，避免内存抖动)
	cv::Mat s_flat = src.reshape(1, src.total());
	cv::Mat r_flat = ref.reshape(1, ref.total());
	cv::Mat s_64, r_64;
	s_flat.convertTo(s_64, CV_64F);
	r_flat.convertTo(r_64, CV_64F);

	// 2. 直接从 64F 矩阵计算统计量
	// 【修改】删除了原有的 computeMeanCov 调用，直接利用已有的 64F 数据计算
	cv::Mat mu_s, cov_s, mu_r, cov_r;
	cv::calcCovarMatrix(s_64, cov_s, mu_s, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
	cv::calcCovarMatrix(r_64, cov_r, mu_r, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);

	// 3. 计算解析解矩阵 M
	// 【核心修复】使用 cv::repeat 解决崩溃问题，确保减法操作尺寸一致
	cv::Mat s_centered = s_64 - cv::repeat(mu_s, s_64.rows, 1);
	cv::Mat r_centered = r_64 - cv::repeat(mu_r, r_64.rows, 1);

	cv::Mat cov_r_inv;
	cv::invert(cov_r, cov_r_inv, cv::DECOMP_SVD);

	// A = (ref - mu_ref) * inv(cov_ref)
	cv::Mat A = r_centered * cov_r_inv;

	// 求解最小二乘 (等价于 Python 的 pinv 逻辑): X = pinv(A) * s_centered
	cv::Mat X;
	cv::solve(A.t() * A, A.t() * s_centered, X, cv::DECOMP_SVD);

	cv::Mat cov_s_inv;
	cv::invert(cov_s, cov_s_inv, cv::DECOMP_SVD);

	// T = (X * inv(cov_src)).T
	cv::Mat T = (X * cov_s_inv).t();

	// 4. 应用变换
	// 变换公式：P_out = P_in * T + (mu_ref - mu_src * T)
	cv::Mat B = mu_r - mu_s * T;

	cv::Mat T_f, B_f, Trans;
	T.convertTo(T_f, CV_32F);
	B.convertTo(B_f, CV_32F);

	// 构造 3x4 矩阵 M = [T^T | B^T]，用于 cv::transform
	cv::hconcat(T_f.t(), B_f.t(), Trans);

	cv::transform(src, src, Trans);
}

// =========================================================
// Reinhard Implementation
// =========================================================

void ColorEqualize::reinhard(cv::Mat &src, const cv::Scalar &ref_mean, const cv::Scalar &ref_std) {
	// 【核心修复】前置条件检查，不满足 8-bit RGB 则直接返回，避免浪费计算资源
	if (src.empty() || src.type() != CV_8UC3) {
		return;
	}

	cv::Mat processed;
	// 简化：固定为 1/255.0f
	src.convertTo(processed, CV_32F, 1.0f / 255.0f);

	cv::Scalar src_mean, src_std;
	cv::meanStdDev(processed, src_mean, src_std);

	std::vector<cv::Mat> channels;
	cv::split(processed, channels);

	for (int i = 0; i < 3; ++i) {
		double s_std = std::max(src_std[i], 1e-6); // 防止除零
		double alpha = ref_std[i] / s_std;
		double beta = ref_mean[i] - alpha * src_mean[i];

		// 线性映射：I_out = alpha * I_in + beta
		channels[i].convertTo(channels[i], -1, alpha, beta);

		// 钳位处理：RGB 归一化空间必须在 [0.0, 1.0]
		cv::threshold(channels[i], channels[i], 1.0, 1.0, cv::THRESH_TRUNC);
		cv::threshold(channels[i], channels[i], 0.0, 0.0, cv::THRESH_TOZERO);
	}

	cv::merge(channels, processed);

	// 恢复位深：固定恢复到 CV_8U
	processed.convertTo(src, CV_8U, 255.0f);
}

void ColorEqualize::computeLabStats(const cv::Mat &src, cv::Scalar &mean, cv::Scalar &stddev) {
	// 安全检查：仅支持 8-bit RGB
	if (src.empty() || src.type() != CV_8UC3) {
		mean = cv::Scalar::all(0);
		stddev = cv::Scalar::all(0);
		return;
	}

	cv::Mat temp;
	src.convertTo(temp, CV_32F, 1.0f / 255.0f);
	cv::meanStdDev(temp, mean, stddev);
}

// =========================================================
// HistMatch Implementation
// =========================================================

void ColorEqualize::histMatch(cv::Mat &src, const cv::Mat &ref) {
	// 1. 类型安全检查：不满足 8-bit RGB 则直接返回，减少冗余开销
	if (src.empty() || ref.empty() || src.type() != CV_8UC3 || ref.type() != CV_8UC3) {
		return;
	}

	// 2. 分离通道：Python 代码中是 for ch in range(p) 处理所有通道
	std::vector<cv::Mat> s_chans, r_chans;
	cv::split(src, s_chans);
	cv::split(ref, r_chans);

	// 对 RGB 三个通道分别进行匹配
	for (int i = 0; i < 3; ++i) {
		histMatchChannel(s_chans[i], r_chans[i]);
	}

	// 3. 合并回原图
	cv::merge(s_chans, src);
}

void ColorEqualize::histMatchChannel(cv::Mat &src, const cv::Mat &ref) {
	// 8-bit 图像固定 256 级灰度
	const int histSize = 256;
	float range[] = {0, 256};
	const float *histRange = {range};
	cv::Mat s_hist, r_hist;

	// 1. 计算直方图
	cv::calcHist(&src, 1, 0, cv::Mat(), s_hist, 1, &histSize, &histRange);
	cv::calcHist(&ref, 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);

	// 2. 计算累积分布函数 (CDF)
	float s_total = (float)src.total();
	float r_total = (float)ref.total();
	std::vector<float> s_cdf(256), r_cdf(256);

	s_cdf[0] = s_hist.at<float>(0) / s_total;
	r_cdf[0] = r_hist.at<float>(0) / r_total;
	for (int i = 1; i < 256; ++i) {
		s_cdf[i] = s_cdf[i - 1] + (s_hist.at<float>(i) / s_total);
		r_cdf[i] = r_cdf[i - 1] + (r_hist.at<float>(i) / r_total);
	}

	// 3. 构建查找表 (LUT)
	// 匹配逻辑：对于 src 的每个灰度 i，寻找 ref 中 CDF 最接近的灰度 j
	cv::Mat lut(1, 256, CV_8U);
	uchar *p_lut = lut.ptr<uchar>();
	int last_j = 0;
	for (int i = 0; i < 256; ++i) {
		while (last_j < 255 && r_cdf[last_j] < s_cdf[i]) {
			last_j++;
		}
		p_lut[i] = (uchar)last_j;
	}

	// 4. 性能优化：直接使用 OpenCV 的 SIMD 加速 LUT 函数，替代手动循环
	cv::LUT(src, lut, src);
}
