#include "colormatcher.h"

#include <cmath>
#include <omp.h> // 确保你的编译器开启了 OpenMP
#include <vector>

void ColorMatcher::equalize(std::vector<cv::Mat> &views, Method method) {
	int num_views = static_cast<int>(views.size());

	// 1. 确定中心视角索引 (Reference)
	int side_len = static_cast<int>(std::sqrt(num_views));
	int center_idx = (side_len / 2) * side_len + (side_len / 2);
	if (center_idx >= num_views)
		center_idx = 0;

	const cv::Mat &ref_img = views[center_idx];

	// 2. 根据不同算法进行批处理优化
	if (method == Method::Reinhard) {
		// 【优化】Reinhard 只需要计算一次参考图的均值和方差
		cv::Scalar ref_mean, ref_std;
		computeLabStats(ref_img, ref_mean, ref_std);

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_views; ++i) {
			if (i == center_idx)
				continue;
			// 调用内部优化接口，避免重复计算 ref 统计量
			reinhardInternal(views[i], ref_mean, ref_std);
		}
	} else if (method == Method::HistMatch) {
// HistMatch 需要针对每张图做匹配 (参考图较大时，copy开销可忽略，直接传引用)
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_views; ++i) {
			if (i == center_idx)
				continue;
			histMatch(views[i], ref_img);
		}
	}
}

void ColorMatcher::apply(cv::Mat &src, const cv::Mat &ref, Method method) {
	if (method == Method::Reinhard) {
		reinhard(src, ref);
	} else if (method == Method::HistMatch) {
		histMatch(src, ref);
	}
}

// =========================================================
// Reinhard Implementation
// =========================================================

void ColorMatcher::reinhard(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty())
		return;

	cv::Scalar ref_mean, ref_std;
	computeLabStats(ref, ref_mean, ref_std);
	reinhardInternal(src, ref_mean, ref_std);
}

void ColorMatcher::computeLabStats(const cv::Mat &src, cv::Scalar &mean,
								   cv::Scalar &stddev) {
	cv::Mat lab;
	int depth = src.depth();

	// 统一转换为 CV_32F 并归一化到 0-1 范围以便转换 Lab
	if (depth == CV_8U) {
		src.convertTo(lab, CV_32F, 1.0 / 255.0);
	} else if (depth == CV_16U) {
		src.convertTo(lab, CV_32F, 1.0 / 65535.0);
	} else if (depth == CV_32F || depth == CV_64F) {
		lab = src; // 已经是浮点数，假设已经是归一化的或者直接可用
	} else {
		src.convertTo(lab, CV_32F);
	}

	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	cv::meanStdDev(lab, mean, stddev);
}

void ColorMatcher::reinhardInternal(cv::Mat &src, const cv::Scalar &ref_mean,
									const cv::Scalar &ref_std) {
	if (src.empty())
		return;

	cv::Mat lab;
	int original_depth = src.depth();

	// 1. 转 Lab
	if (original_depth == CV_8U) {
		src.convertTo(lab, CV_32F, 1.0 / 255.0);
	} else if (original_depth == CV_16U) {
		src.convertTo(lab, CV_32F, 1.0 / 65535.0);
	} else {
		src.convertTo(lab, CV_32F);
	}

	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);

	// 2. 计算源图像统计量
	cv::Scalar src_mean, src_std;
	cv::meanStdDev(lab, src_mean, src_std);

	// 3. 通道独立变换
	std::vector<cv::Mat> channels;
	cv::split(lab, channels);

	for (int i = 0; i < 3; ++i) {
		// 避免除以零
		double s_std = (src_std[i] < 1e-6) ? 1e-6 : src_std[i];

		// Reinhard 核心公式: res = (pixel - mean_src) * (std_ref / std_src) +
		// mean_ref
		double alpha = ref_std[i] / s_std;
		double beta = ref_mean[i] - alpha * src_mean[i];

		// convertTo 支持线性变换: dst = src * alpha + beta
		channels[i].convertTo(channels[i], -1, alpha, beta);
	}

	// 4. 转回 BGR
	cv::merge(channels, lab);
	cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);

	// 5. 恢复原始位深
	if (original_depth == CV_8U) {
		lab.convertTo(src, CV_8U, 255.0);
	} else if (original_depth == CV_16U) {
		lab.convertTo(src, CV_16U, 65535.0);
	} else {
		lab.copyTo(src);
	}
}

// =========================================================
// Histogram Matching Implementation
// =========================================================

void ColorMatcher::histMatch(cv::Mat &src, const cv::Mat &ref) {
	if (src.empty() || ref.empty())
		return;
	if (src.channels() != ref.channels())
		return;

	std::vector<cv::Mat> src_channels, ref_channels;
	cv::split(src, src_channels);
	cv::split(ref, ref_channels);

	// 假设是 3 通道图像
	for (size_t i = 0; i < src_channels.size(); ++i) {
		histMatchChannel(src_channels[i], ref_channels[i]);
	}
	cv::merge(src_channels, src);
}

void ColorMatcher::histMatchChannel(cv::Mat &src, const cv::Mat &ref) {
	// 简化处理：先把数据转为 8U 计算直方图，对于 HDR 图像可能需要更复杂的
	// Quantization
	cv::Mat src_8u, ref_8u;
	if (src.depth() != CV_8U)
		src.convertTo(src_8u, CV_8U, 255.0);
	else
		src_8u = src;

	if (ref.depth() != CV_8U)
		ref.convertTo(ref_8u, CV_8U, 255.0);
	else
		ref_8u = ref;

	int histSize = 256;
	float range[] = {0, 256};
	const float *histRange = {range};

	cv::Mat src_hist, ref_hist;
	cv::calcHist(&src_8u, 1, 0, cv::Mat(), src_hist, 1, &histSize, &histRange,
				 true, false);
	cv::calcHist(&ref_8u, 1, 0, cv::Mat(), ref_hist, 1, &histSize, &histRange,
				 true, false);

	cv::normalize(src_hist, src_hist, 1, 0, cv::NORM_L1);
	cv::normalize(ref_hist, ref_hist, 1, 0, cv::NORM_L1);

	cv::Mat src_cdf(histSize, 1, CV_32F);
	cv::Mat ref_cdf(histSize, 1, CV_32F);

	src_cdf.at<float>(0) = src_hist.at<float>(0);
	ref_cdf.at<float>(0) = ref_hist.at<float>(0);

	for (int i = 1; i < histSize; ++i) {
		src_cdf.at<float>(i) = src_cdf.at<float>(i - 1) + src_hist.at<float>(i);
		ref_cdf.at<float>(i) = ref_cdf.at<float>(i - 1) + ref_hist.at<float>(i);
	}

	cv::Mat lut(1, 256, CV_8U);
	for (int i = 0; i < histSize; ++i) {
		float src_val = src_cdf.at<float>(i);
		int best_j = 0;
		float min_diff = 1.0f;
		for (int j = 0; j < histSize; ++j) {
			float diff = std::abs(src_val - ref_cdf.at<float>(j));
			if (diff < min_diff) {
				min_diff = diff;
				best_j = j;
			}
		}
		lut.at<uchar>(i) = (uchar)best_j;
	}

	// 应用 LUT (注意：LUT 只适用于 8U)
	// 如果原图是 float，我们需要将 float -> 8U -> LUT -> float 的过程，
	// 这里为了简单，我们直接修改 src (如果是 float，精度会损失到 8bit)
	// 实际工程中，对于 float 图像通常不做直方图匹配，或者需要 65536 级的 LUT

	if (src.depth() == CV_8U) {
		cv::LUT(src, lut, src);
	} else {
		// 如果原图不是 8U，先 LUT 8U 版本，再转回去 (会丢失精度)
		cv::Mat temp;
		cv::LUT(src_8u, lut, temp);
		if (src.depth() == CV_32F)
			temp.convertTo(src, CV_32F, 1.0 / 255.0);
		else if (src.depth() == CV_16U)
			temp.convertTo(src, CV_16U, 65535.0 / 255.0);
	}
}