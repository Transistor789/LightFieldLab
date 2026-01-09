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
	// 展平为 Nx3 矩阵
	cv::Mat reshaped = img.reshape(1, img.total());

	// 转为 CV_64F 保证精度
	cv::Mat float_img;
	reshaped.convertTo(float_img, CV_64F);

	// 计算协方差 (SCALE 标志会自动除以 N)
	cv::calcCovarMatrix(float_img, cov, mean,
						cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
}

// 辅助：计算矩阵平方根 A^(1/2)
cv::Mat ColorMatcher::sqrtMatrix(const cv::Mat &cov) {
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	// 构造对角矩阵 sqrt(Lambda)
	cv::Mat sqrt_lambda = cv::Mat::zeros(3, 3, CV_64F);
	for (int i = 0; i < 3; ++i) {
		double val = eigenvalues.at<double>(i);
		sqrt_lambda.at<double>(i, i) = (val > 0) ? std::sqrt(val) : 0.0;
	}

	// A^(1/2) = V * sqrt(D) * V^T
	return eigenvectors.t() * sqrt_lambda * eigenvectors;
}

// 辅助：计算矩阵逆平方根 A^(-1/2)
cv::Mat ColorMatcher::invSqrtMatrix(const cv::Mat &cov) {
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat inv_sqrt_lambda = cv::Mat::zeros(3, 3, CV_64F);
	for (int i = 0; i < 3; ++i) {
		double val = eigenvalues.at<double>(i);
		inv_sqrt_lambda.at<double>(i, i) =
			(val > 1e-9) ? (1.0 / std::sqrt(val)) : 0.0;
	}

	return eigenvectors.t() * inv_sqrt_lambda * eigenvectors;
}

void ColorMatcher::mkl(cv::Mat &src, const cv::Mat &ref) {
	// 1. 准备数据
	int original_depth = src.depth();
	cv::Mat src_64, ref_64;

	// 转为 CV_64F 进行计算
	if (src.channels() != 3 || ref.channels() != 3)
		return;

	// Reshape to Nx3
	cv::Mat r = src.reshape(1, src.total());
	r.convertTo(src_64, CV_64F);

	// 2. 计算统计量
	cv::Mat mu_r, cov_r, mu_z, cov_z;
	computeMeanCov(src, mu_r, cov_r);
	computeMeanCov(ref, mu_z, cov_z);

	// 3. 计算 MKL 变换矩阵 T
	// T = Cov_r^(-1/2) * (Cov_r^(1/2) * Cov_z * Cov_r^(1/2))^(1/2) *
	// Cov_r^(-1/2)

	cv::Mat cov_r_sqrt = sqrtMatrix(cov_r);
	cv::Mat cov_r_inv_sqrt = invSqrtMatrix(cov_r);

	cv::Mat C = cov_r_sqrt * cov_z * cov_r_sqrt;
	cv::Mat C_sqrt = sqrtMatrix(C);

	cv::Mat T = cov_r_inv_sqrt * C_sqrt * cov_r_inv_sqrt;

	// 4. 应用变换: Res = (Src - mu_r) * T + mu_z
	// 注意 OpenCV 行向量乘法: Mat * T
	cv::Mat centered = src_64 - cv::repeat(mu_r, src_64.rows, 1);
	cv::Mat result = centered * T + cv::repeat(mu_z, src_64.rows, 1);

	// 5. 恢复形状和类型
	cv::Mat result_reshaped = result.reshape(3, src.rows);
	result_reshaped.convertTo(src, original_depth);
}

void ColorMatcher::mvgd(cv::Mat &src, const cv::Mat &ref) {
	// 1. 准备数据
	int original_depth = src.depth();
	cv::Mat src_64, ref_64;

	if (src.channels() != 3 || ref.channels() != 3)
		return;

	// 如果尺寸不一致，MVGD 解析解所需的对应关系不存在，降级或返回
	if (src.size() != ref.size()) {
		// Fallback to MKL if sizes mismatch (MVGD requires spatial
		// correspondence logic)
		mkl(src, ref);
		return;
	}

	// Reshape Nx3
	cv::Mat r = src.reshape(1, src.total());
	cv::Mat z = ref.reshape(1, ref.total());
	r.convertTo(src_64, CV_64F);
	z.convertTo(ref_64, CV_64F);

	// 2. 计算统计量
	cv::Mat mu_r, cov_r, mu_z, cov_z;
	computeMeanCov(src, mu_r, cov_r);
	computeMeanCov(ref, mu_z, cov_z);

	cv::Mat cov_r_inv, cov_z_inv;
	cv::invert(cov_r, cov_r_inv);
	cv::invert(cov_z, cov_z_inv);

	// 3. 计算 MVGD 变换矩阵 (Analytical Solver)
	// Python Logic: pinv( (Z-mu_z) * cov_z_inv ) * (R-mu_r) * cov_r_inv
	// Let A = (Z-mu_z) * cov_z_inv
	// Let B = (R-mu_r)
	// We need: (A^+ * B) * cov_r_inv
	// Calculation: A^+ B = (A^T A)^(-1) A^T B

	cv::Mat z_centered = ref_64 - cv::repeat(mu_z, ref_64.rows, 1);
	cv::Mat r_centered = src_64 - cv::repeat(mu_r, src_64.rows, 1);

	// A = Z_c * Cov_z^(-1)
	cv::Mat A = z_centered * cov_z_inv;

	// 为了节省内存，不直接存储大矩阵 A，而是通过累加计算 A^T*A 和 A^T*B
	// A^T * A (3x3)
	// A^T * B (3x3)
	cv::Mat AtA, AtB;
	cv::mulTransposed(A, AtA, true); // AtA = A^T * A
	AtB = A.t() * r_centered;		 // AtB = A^T * B (Mat Mul)

	// X = (A^T A)^(-1) * (A^T B)
	cv::Mat AtA_inv;
	cv::invert(AtA, AtA_inv);
	cv::Mat X = AtA_inv * AtB;

	// T = (X * cov_r_inv)^T = cov_r_inv^T * X^T = cov_r_inv * X^T (Symmetric)
	// Python code returns matrix applied on RIGHT side of row vectors.
	// Transpose required at end of python derivation?
	// Python: return dot(..., cov_r_inv).T
	// C++ Matrix * T:
	// M = (X * cov_r_inv).t();
	cv::Mat M = (X * cov_r_inv).t();

	// 4. 应用变换
	cv::Mat result = r_centered * M + cv::repeat(mu_z, src_64.rows, 1);

	// 5. 恢复
	cv::Mat result_reshaped = result.reshape(3, src.rows);
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

	// 1. 转 Lab
	int original_depth = src.depth();
	cv::Mat lab;
	if (original_depth == CV_8U)
		src.convertTo(lab, CV_32F, 1.0 / 255.0);
	else if (original_depth == CV_16U)
		src.convertTo(lab, CV_32F, 1.0 / 65535.0);
	else
		src.convertTo(lab, CV_32F);

	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);

	// 2. 统计当前图
	cv::Scalar src_mean, src_std;
	cv::meanStdDev(lab, src_mean, src_std);

	// 3. 应用变换 (逐通道)
	std::vector<cv::Mat> channels;
	cv::split(lab, channels);

	for (int i = 0; i < 3; ++i) {
		double s_std = (src_std[i] < 1e-6) ? 1e-6 : src_std[i];
		double alpha = ref_std[i] / s_std;
		double beta = ref_mean[i] - alpha * src_mean[i];
		channels[i].convertTo(channels[i], -1, alpha, beta);
	}

	cv::merge(channels, lab);
	cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);

	// 4. 恢复位深
	if (original_depth == CV_8U)
		lab.convertTo(src, CV_8U, 255.0);
	else if (original_depth == CV_16U)
		lab.convertTo(src, CV_16U, 65535.0);
	else
		lab.copyTo(src);
}

void ColorMatcher::computeLabStats(const cv::Mat &src, cv::Scalar &mean,
								   cv::Scalar &stddev) {
	cv::Mat lab;
	int d = src.depth();
	if (d == CV_8U)
		src.convertTo(lab, CV_32F, 1.0 / 255.0);
	else if (d == CV_16U)
		src.convertTo(lab, CV_32F, 1.0 / 65535.0);
	else
		src.convertTo(lab, CV_32F);

	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	cv::meanStdDev(lab, mean, stddev);
}

// =========================================================
// HistMatch Implementation
// =========================================================

void ColorMatcher::histMatch(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty())
		return;

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