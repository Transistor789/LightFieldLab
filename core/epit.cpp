#include "epit.h"

#include <algorithm>
#include <iostream>
#include <omp.h> // OpenMP

std::vector<cv::Mat> EPIT::infer(const std::vector<cv::Mat> &yViews) {
	int H = yViews[0].rows;
	int W = yViews[0].cols;
	int outH = H * scale_;
	int outW = W * scale_;

	// A. 初始化结果容器
	int totalViews = center_only_ ? 1 : (ang_res_ * ang_res_);
	std::vector<cv::Mat> outputResults(totalViews);
	for (int i = 0; i < totalViews; ++i) {
		outputResults[i] = cv::Mat::zeros(outH, outW, CV_8UC1);
	}

	// B. 参数准备
	int step = patch_size_ - 2 * padding_;

	// EPIT 输入布局是 Stack: [Batch, View, PatchH, PatchW]
	size_t singlePatchPixels = patch_size_ * patch_size_;
	size_t inputLen = ang_res_ * ang_res_ * singlePatchPixels;

	// 输出布局
	size_t outPatchH = patch_size_ * scale_;
	size_t outPatchW = patch_size_ * scale_;
	size_t singleOutPatchPixels = outPatchH * outPatchW;
	size_t outLen = ang_res_ * ang_res_ * singleOutPatchPixels;

	// Buffer (循环外分配)
	std::vector<float> inputBuffer(inputLen);
	std::vector<float> outputBuffer(outLen);

	// C. 空间循环
	for (int y = 0; y < H; y += step) {
		for (int x = 0; x < W; x += step) {
			// --- 计算 ROI ---
			int x_start = std::max(0, x - padding_);
			int y_start = std::max(0, y - padding_);
			int x_end_req = x_start + patch_size_;
			int y_end_req = y_start + patch_size_;

			int read_w = std::min(W, x_end_req) - x_start;
			int read_h = std::min(H, y_end_req) - y_start;

			// --- 填充 Input Buffer (Stack Layout) ---
			std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
			float *bufferBase = inputBuffer.data();

			// 注意：EPIT 的填充逻辑是按 View 连续存储
			for (int u = 0; u < ang_res_; ++u) {
				for (int v = 0; v < ang_res_; ++v) {
					int viewIdx = u * ang_res_ + v;
					cv::Mat patch = yViews[viewIdx](
						cv::Rect(x_start, y_start, read_w, read_h));

					float *viewPtr = bufferBase + viewIdx * singlePatchPixels;
					for (int r = 0; r < read_h; ++r) {
						float *rowPtr =
							viewPtr + r * patch_size_; // 指向当前行的起始
						memcpy(rowPtr, patch.ptr<float>(r),
							   read_w * sizeof(float));
					}
				}
			}

			// --- 推理 ---
			net_->Infer(inputBuffer.data(), outputBuffer.data());

			// --- 提取输出并拼接 ---
			float *outBase = outputBuffer.data();

			// 公共拼接坐标
			int valid_x_src = (x == 0) ? 0 : padding_;
			int valid_y_src = (y == 0) ? 0 : padding_;
			int actual_w = std::min(step, W - x);
			int actual_h = std::min(step, H - y);

			if (actual_w <= 0 || actual_h <= 0)
				continue;

			cv::Rect srcROI(valid_x_src * scale_, valid_y_src * scale_,
							actual_w * scale_, actual_h * scale_);
			cv::Rect dstROI(x * scale_, y * scale_, actual_w * scale_,
							actual_h * scale_);

			if (center_only_) {
				// === 只提取中心视角 ===
				int centerViewIdx = (ang_res_ / 2) * ang_res_ + (ang_res_ / 2);
				float *patchPtr =
					outBase + centerViewIdx * singleOutPatchPixels;

				cv::Mat outPatchMat(outPatchH, outPatchW, CV_32F, patchPtr);

				// 转 8U
				cv::Mat srPatch8U;
				outPatchMat.convertTo(srPatch8U, CV_8U, 255.0);

				srPatch8U(srcROI).copyTo(outputResults[0](dstROI));

			} else {
				// === 提取所有视角 ===
				for (int u = 0; u < ang_res_; ++u) {
					for (int v = 0; v < ang_res_; ++v) {
						int viewIdx = u * ang_res_ + v;

						// 假设输出也是 Stack 布局: [View][H][W]
						float *patchPtr =
							outBase + viewIdx * singleOutPatchPixels;

						cv::Mat outPatchMat(outPatchH, outPatchW, CV_32F,
											patchPtr);

						// 转 8U
						cv::Mat srPatch8U;
						outPatchMat.convertTo(srPatch8U, CV_8U, 255.0);

						srPatch8U(srcROI).copyTo(
							outputResults[viewIdx](dstROI));
					}
				}
			}
		}
	}
	return outputResults;
}
