#include "distgssr.h"

#include <algorithm>
#include <iostream>
#include <omp.h> // OpenMP 支持

std::vector<cv::Mat> DistgSSR::infer(const std::vector<cv::Mat> &yViews) {
	int H = yViews[0].rows;
	int W = yViews[0].cols;
	int outH = H * scale_;
	int outW = W * scale_;

	// 1. 初始化结果容器
	// 根据 center_only_ 决定输出数量
	int totalViews = center_only_ ? 1 : (ang_res_ * ang_res_);
	std::vector<cv::Mat> outputResults(totalViews);

	for (int i = 0; i < totalViews; ++i) {
		outputResults[i] = cv::Mat::zeros(outH, outW, CV_8UC1);
	}

	// 参数准备
	int step = patch_size_ - 2 * padding_;
	int miniMosaicSize = ang_res_ * patch_size_;
	int outMiniMosaicSize = miniMosaicSize * scale_;

	// Buffer (在循环外分配，避免重复 malloc)
	std::vector<float> inputBuffer(miniMosaicSize * miniMosaicSize);
	std::vector<float> outputBuffer(outMiniMosaicSize * outMiniMosaicSize);

	// === 空间分块循环 (Spatial Loop) ===
	// 注意：此处不建议使用 OpenMP，因为 TensorRT Context 通常不是线程安全的
	for (int y = 0; y < H; y += step) {
		for (int x = 0; x < W; x += step) {
			// --- A. 计算输入 ROI ---
			int x_start = std::max(0, x - padding_);
			int y_start = std::max(0, y - padding_);
			int x_end_req = x_start + patch_size_;
			int y_end_req = y_start + patch_size_;

			int read_w = std::min(W, x_end_req) - x_start;
			int read_h = std::min(H, y_end_req) - y_start;

			// --- B. 构建输入 Mosaic ---
			std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
			float *tensorBasePtr = inputBuffer.data();

			for (int u = 0; u < ang_res_; ++u) {
				for (int v = 0; v < ang_res_; ++v) {
					int viewIdx = u * ang_res_ + v;
					cv::Mat patch = yViews[viewIdx](
						cv::Rect(x_start, y_start, read_w, read_h));

					// 逐行拷贝 (Memory Copy Optimization)
					for (int r = 0; r < read_h; ++r) {
						float *dstPtr = tensorBasePtr
										+ (u * patch_size_ + r) * miniMosaicSize
										+ (v * patch_size_);
						memcpy(dstPtr, patch.ptr<float>(r),
							   read_w * sizeof(float));
					}
				}
			}

			// --- C. 执行推理 ---
			net_->Infer(inputBuffer.data(), outputBuffer.data());

			// --- D. 解析输出 ---
			cv::Mat outMiniMat(outMiniMosaicSize, outMiniMosaicSize, CV_32F,
							   outputBuffer.data());

			// --- E. 计算拼接坐标 (Common) ---
			int valid_x_src_offset = (x == 0) ? 0 : padding_;
			int valid_y_src_offset = (y == 0) ? 0 : padding_;
			int actual_step_w = std::min(step, W - x);
			int actual_step_h = std::min(step, H - y);

			if (actual_step_w <= 0 || actual_step_h <= 0)
				continue;

			cv::Rect srcROI(valid_x_src_offset * scale_,
							valid_y_src_offset * scale_, actual_step_w * scale_,
							actual_step_h * scale_);
			cv::Rect dstROI(x * scale_, y * scale_, actual_step_w * scale_,
							actual_step_h * scale_);

			// --- F. 拼接回大图 ---
			if (center_only_) {
				// 仅提取中心视角
				int center_u = ang_res_ / 2;
				int center_v = ang_res_ / 2;

				int roiY = center_u * (patch_size_ * scale_);
				int roiX = center_v * (patch_size_ * scale_);

				cv::Mat srPatch = outMiniMat(cv::Rect(
					roiX, roiY, patch_size_ * scale_, patch_size_ * scale_));

				// 转 8U (0-255)
				cv::Mat srPatch8U;
				srPatch.convertTo(srPatch8U, CV_8U, 255.0);

				srPatch8U(srcROI).copyTo(outputResults[0](dstROI));

			} else {
				// 提取所有视角
				for (int u = 0; u < ang_res_; ++u) {
					for (int v = 0; v < ang_res_; ++v) {
						int roiY = u * (patch_size_ * scale_);
						int roiX = v * (patch_size_ * scale_);

						cv::Mat srPatch = outMiniMat(
							cv::Rect(roiX, roiY, patch_size_ * scale_,
									 patch_size_ * scale_));

						// 转 8U (0-255)
						cv::Mat srPatch8U;
						srPatch.convertTo(srPatch8U, CV_8U, 255.0);

						int viewIdx = u * ang_res_ + v;
						srPatch8U(srcROI).copyTo(
							outputResults[viewIdx](dstROI));
					}
				}
			}
		}
	}

	return outputResults;
}
