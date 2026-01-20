#include "colormatcher.h"

#include <cmath>
#include <omp.h>
#include <vector>

// =========================================================
// Public Interface
// =========================================================

void ColorMatcher::equalize(std::vector<cv::Mat> &views,
							ColorEqualizeMethod method) {
	if (views.empty())
		return;

	int num_views = static_cast<int>(views.size());

	// 1. 确定中心视角索引 (Reference)
	int side_len = static_cast<int>(std::sqrt(num_views));
	int center_idx = (side_len / 2) * side_len + (side_len / 2);
	if (center_idx >= num_views)
		center_idx = 0;

	const cv::Mat &ref_img = views[center_idx];

	// 2. 根据不同算法进行批处理优化
	if (method == ColorEqualizeMethod::Reinhard) {
		cv::Scalar ref_mean, ref_std;
		computeLabStats(ref_img, ref_mean, ref_std);

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_views; ++i) {
			if (i == center_idx)
				continue;
			reinhardInternal(views[i], ref_mean, ref_std);
		}
	} else {
		// 对于 HistMatch, MKL, MVGD 等方法，通常需要完整的 ref 图像信息
		// 且计算量主要在单图变换，简单并行即可
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_views; ++i) {
			if (i == center_idx)
				continue;
			apply(views[i], ref_img, method);
		}
	}
}

void ColorMatcher::apply(cv::Mat &src, const cv::Mat &ref,
						 ColorEqualizeMethod method) {
	if (src.empty() || ref.empty())
		return;

	switch (method) {
		case ColorEqualizeMethod::Reinhard:
			reinhard(src, ref);
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
void ColorMatcher::computeMeanCov(const cv::Mat &img, cv::Mat &mean,
								  cv::Mat &cov) {
	int chans = img.channels();
	cv::Mat reshaped = img.reshape(1, img.total());
	cv::Mat float_img;
	reshaped.convertTo(float_img, CV_64F);

	// 计算协方差矩阵，维度自动适配通道数 (1x1 或 3x3)
	cv::calcCovarMatrix(float_img, cov, mean,
						cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
}

// 辅助：计算矩阵平方根 A^(1/2)
cv::Mat ColorMatcher::sqrtMatrix(const cv::Mat &cov) {
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
cv::Mat ColorMatcher::invSqrtMatrix(const cv::Mat &cov) {
	int n = cov.rows; // 动态获取通道数
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat inv_sqrt_lambda = cv::Mat::zeros(n, n, CV_64F);
	for (int i = 0; i < n; ++i) {
		double val = eigenvalues.at<double>(i);
		inv_sqrt_lambda.at<double>(i, i) =
			(val > 1e-9) ? (1.0 / std::sqrt(val)) : 0.0;
	}
	return eigenvectors.t() * inv_sqrt_lambda * eigenvectors;
}

void ColorMatcher::mkl(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty())
		return;
	int original_depth = src.depth();
	int chans = src.channels();

	// Reshape 为 NxChans
	cv::Mat src_flat = src.reshape(1, src.total());
	cv::Mat src_64;
	src_flat.convertTo(src_64, CV_64F);

	// 2. 计算统计量
	cv::Mat mu_r, cov_r, mu_z, cov_z;
	computeMeanCov(src, mu_r, cov_r);
	computeMeanCov(ref, mu_z, cov_z);

	// 3. 计算 MKL 变换矩阵 T (即使是 1x1 矩阵，线性代数逻辑依然成立)
	cv::Mat cov_r_sqrt = sqrtMatrix(cov_r);
	cv::Mat cov_r_inv_sqrt = invSqrtMatrix(cov_r);
	cv::Mat C = cov_r_sqrt * cov_z * cov_r_sqrt;
	cv::Mat C_sqrt = sqrtMatrix(C);
	cv::Mat T = cov_r_inv_sqrt * C_sqrt * cov_r_inv_sqrt;

	// 4. 应用变换
	cv::Mat centered = src_64 - cv::repeat(mu_r, src_64.rows, 1);
	cv::Mat result = centered * T + cv::repeat(mu_z, src_64.rows, 1);

	// 数值截断：防止低光增强后出现大片白色死区
	double max_val = (original_depth == CV_16U) ? 65535.0 : 255.0;
	cv::max(0.0, cv::min(max_val, result), result);

	// 5. 恢复形状
	cv::Mat result_reshaped = result.reshape(chans, src.rows);
	result_reshaped.convertTo(src, original_depth);
}

void ColorMatcher::mvgd(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty())
		return;

	int original_depth = src.depth();
	int chans = src.channels();
	cv::Size src_size = src.size();

	// 1. 降级处理：如果尺寸不一致，MVGD 的空间对应逻辑无法执行，降级为 MKL
	if (src_size != ref.size()) {
		mkl(src, ref); // MKL 不需要尺寸一致
		return;
	}

	// 2. 数据准备：转为 64F 保证计算精度
	cv::Mat src_64, ref_64;
	src.reshape(1, src.total()).convertTo(src_64, CV_64F);
	ref.reshape(1, ref.total()).convertTo(ref_64, CV_64F);

	// 3. 计算统计量
	cv::Mat mu_r, cov_r, mu_z, cov_z;
	computeMeanCov(src, mu_r, cov_r);
	computeMeanCov(ref, mu_z, cov_z);

	cv::Mat cov_r_inv, cov_z_inv;
	// 使用 DECOMP_SVD 提高求逆的鲁棒性
	cv::invert(cov_r, cov_r_inv, cv::DECOMP_SVD);
	cv::invert(cov_z, cov_z_inv, cv::DECOMP_SVD);

	// 4. 计算 MVGD 变换矩阵 (针对 NxC 矩阵的解析解)
	cv::Mat r_centered = src_64 - cv::repeat(mu_r, src_64.rows, 1);
	cv::Mat z_centered = ref_64 - cv::repeat(mu_z, ref_64.rows, 1);

	// A = Z_c * Cov_z^(-1)
	cv::Mat A = z_centered * cov_z_inv;

	// 计算 X = (A^T * A)^(-1) * (A^T * r_centered)
	cv::Mat AtA, AtB;
	cv::mulTransposed(A, AtA, true);
	AtB = A.t() * r_centered;

	cv::Mat AtA_inv;
	cv::invert(AtA, AtA_inv, cv::DECOMP_SVD);
	cv::Mat X = AtA_inv * AtB;

	// 最终变换矩阵 M = (X * cov_r_inv)^T
	cv::Mat M = (X * cov_r_inv).t();

	// 5. 应用变换并执行数值截断
	cv::Mat result = r_centered * M + cv::repeat(mu_z, src_64.rows, 1);

	// 根据位深确定截断阈值
	double max_val = (original_depth == CV_16U) ? 65535.0 : 255.0;
	cv::max(0.0, cv::min(max_val, result), result); // 防止低光增强溢出

	// 6. 恢复原始形状与类型
	cv::Mat result_reshaped = result.reshape(chans, src_size.height);
	result_reshaped.convertTo(src, original_depth);
}

// =========================================================
// Reinhard Implementation
// =========================================================

void ColorMatcher::reinhard(cv::Mat &src, const cv::Mat &ref) {
	cv::Scalar ref_mean, ref_std;
	computeLabStats(ref, ref_mean, ref_std);
	reinhardInternal(src, ref_mean, ref_std);
}

void ColorMatcher::reinhardInternal(cv::Mat &src, const cv::Scalar &ref_mean,
									const cv::Scalar &ref_std) {
	if (src.empty())
		return;

	int original_depth = src.depth();
	int chans = src.channels();
	cv::Mat processed;

	float scale_factor =
		(original_depth == CV_16U) ? 1.0f / 65535.0f : 1.0f / 255.0f;
	src.convertTo(processed, CV_32F, scale_factor);

	// 【关键修复】只有 3 通道才转 Lab
	if (chans == 3) {
		cv::cvtColor(processed, processed, cv::COLOR_BGR2Lab);
	}

	cv::Scalar src_mean, src_std;
	cv::meanStdDev(processed, src_mean, src_std);

	std::vector<cv::Mat> channels;
	cv::split(processed, channels);

	for (int i = 0; i < chans; ++i) {
		double s_std = (src_std[i] < 1e-6) ? 1e-6 : src_std[i];
		double alpha = ref_std[i] / s_std;
		double beta = ref_mean[i] - alpha * src_mean[i];

		channels[i].convertTo(channels[i], -1, alpha, beta);

		// --- 修正后的截断逻辑 ---
		if (chans == 3) {
			if (i == 0) {
				// L 通道：0 ~ 100
				cv::max(0.0f, cv::min(100.0f, channels[i]), channels[i]);
			} else {
				// a, b 通道：-128 ~ 127 (不建议截断到 0~1)
				cv::max(-128.0f, cv::min(127.0f, channels[i]), channels[i]);
			}
		} else {
			// 单通道灰度：0 ~ 1
			cv::max(0.0f, cv::min(1.0f, channels[i]), channels[i]);
		}
	}

	cv::merge(channels, processed);

	if (chans == 3) {
		cv::cvtColor(processed, processed, cv::COLOR_Lab2BGR);
	}

	float restore_scale = (original_depth == CV_16U) ? 65535.0f : 255.0f;
	processed.convertTo(src, original_depth, restore_scale);
}

void ColorMatcher::computeLabStats(const cv::Mat &src, cv::Scalar &mean,
								   cv::Scalar &stddev) {
	cv::Mat temp;
	int chans = src.channels();
	float scale = (src.depth() == CV_16U) ? 1.0f / 65535.0f : 1.0f / 255.0f;
	src.convertTo(temp, CV_32F, scale);

	// 【关键修复】单通道不执行 cvtColor
	if (chans == 3) {
		cv::cvtColor(temp, temp, cv::COLOR_BGR2Lab);
	}
	cv::meanStdDev(temp, mean, stddev);
}

// =========================================================
// HistMatch Implementation
// =========================================================

void ColorMatcher::histMatch(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty())
		return;

	int chans = src.channels();
	int original_depth = src.depth();

	if (chans == 1) {
		// 单通道直接匹配
		cv::Mat s_float, r_float;
		src.convertTo(s_float, CV_32F, 1.0 / 255.0);
		ref.convertTo(r_float, CV_32F, 1.0 / 255.0);
		histMatchChannel(s_float, r_float);
		s_float.convertTo(src, original_depth, 255.0);
	} else {
		cv::Mat src_lab, ref_lab;
		int original_depth = src.depth();

		// 转 Lab (32F)
		auto toLab = [](const cv::Mat &in, cv::Mat &out) {
			int d = in.depth();
			if (d == CV_8U)
				in.convertTo(out, CV_32F, 1.0 / 255.0);
			else if (d == CV_16U)
				in.convertTo(out, CV_32F, 1.0 / 65535.0);
			else
				in.convertTo(out, CV_32F);
			cv::cvtColor(out, out, cv::COLOR_BGR2Lab);
		};

		toLab(src, src_lab);
		toLab(ref, ref_lab);

		std::vector<cv::Mat> src_channels, ref_channels;
		cv::split(src_lab, src_channels);
		cv::split(ref_lab, ref_channels);

		// 仅匹配 L 通道 (Index 0)
		histMatchChannel(src_channels[0], ref_channels[0]);

		cv::merge(src_channels, src_lab);
		cv::cvtColor(src_lab, src_lab, cv::COLOR_Lab2BGR);

		if (original_depth == CV_8U)
			src_lab.convertTo(src, CV_8U, 255.0);
		else if (original_depth == CV_16U)
			src_lab.convertTo(src, CV_16U, 65535.0);
		else
			src_lab.copyTo(src);
	}
}

void ColorMatcher::histMatchChannel(cv::Mat &src, const cv::Mat &ref) {
	int histSize = 1024;
	float range[] = {0.0f, 100.0f};
	const float *histRange = {range};

	cv::Mat src_hist, ref_hist;
	cv::calcHist(&src, 1, 0, cv::Mat(), src_hist, 1, &histSize, &histRange,
				 true, false);
	cv::calcHist(&ref, 1, 0, cv::Mat(), ref_hist, 1, &histSize, &histRange,
				 true, false);

	cv::normalize(src_hist, src_hist, 1, 0, cv::NORM_L1);
	cv::normalize(ref_hist, ref_hist, 1, 0, cv::NORM_L1);

	std::vector<float> src_cdf(histSize), ref_cdf(histSize);
	src_cdf[0] = src_hist.at<float>(0);
	ref_cdf[0] = ref_hist.at<float>(0);
	for (int i = 1; i < histSize; ++i) {
		src_cdf[i] = src_cdf[i - 1] + src_hist.at<float>(i);
		ref_cdf[i] = ref_cdf[i - 1] + ref_hist.at<float>(i);
	}

	std::vector<float> lut(histSize);
	for (int i = 0; i < histSize; ++i) {
		float val = src_cdf[i];
		auto it = std::lower_bound(ref_cdf.begin(), ref_cdf.end(), val);
		int idx = std::distance(ref_cdf.begin(), it);
		if (idx >= histSize)
			idx = histSize - 1;
		lut[i] = (float)idx * (100.0f / histSize);
	}

	int rows = src.rows;
	int cols = src.cols;
	if (src.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	float bin_scale = (float)histSize / 100.0f;
	for (int r = 0; r < rows; ++r) {
		float *ptr = src.ptr<float>(r);
		for (int c = 0; c < cols; ++c) {
			float v = ptr[c];
			if (v < 0)
				v = 0;
			if (v > 100)
				v = 100;
			float bin_f = v * bin_scale;
			int bin_i = (int)bin_f;
			if (bin_i >= histSize - 1)
				ptr[c] = lut[histSize - 1];
			else {
				float alpha = bin_f - bin_i;
				ptr[c] = lut[bin_i] * (1.0f - alpha) + lut[bin_i + 1] * alpha;
			}
		}
	}
}