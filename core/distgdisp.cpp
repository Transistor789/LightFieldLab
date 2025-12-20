#include "distgdisp.h"

#include <algorithm>
#include <iostream>
#include <omp.h> // 引入 OpenMP
#include <vector>

// 【建议】在 infer 函数外部或构造函数中调用一次，避免 OpenCV 内部多线程与
// OpenMP 冲突 cv::setNumThreads(0);

cv::Mat DistgDisp::infer(const std::vector<cv::Mat> &grayViews) {
	// 校验
	if (grayViews.size() != ang_res_ * ang_res_) {
		std::cerr << "[DistgDisp] View count mismatch! Expected "
				  << ang_res_ * ang_res_ << std::endl;
		return {};
	}

	int fullH = grayViews[0].rows;
	int fullW = grayViews[0].cols;

	// 1. 结果容器
	cv::Mat fullDisp = cv::Mat::zeros(fullH, fullW, CV_32FC1);

	// 2. 参数计算
	int step = patch_size_ - 2 * padding_;
	int mosaicSize = patch_size_ * ang_res_;

	// 3. Buffer 分配 (使用 vector 保证内存安全)
	size_t inputLen = mosaicSize * mosaicSize;
	size_t outputLen = patch_size_ * patch_size_;

	std::vector<float> inputBuffer(inputLen);
	std::vector<float> outputBuffer(outputLen);

	// 4. 创建 Wrapper (零拷贝技巧)
	// 这个 Mat 不分配内存，直接指向 inputBuffer
	cv::Mat mosaicWrapper(mosaicSize, mosaicSize, CV_32FC1, inputBuffer.data());

	// 5. 空间循环 (必须串行)
	for (int y = 0; y < fullH; y += step) {
		for (int x = 0; x < fullW; x += step) {
			// --- A. 计算 ROI ---
			int x_start = x - padding_;
			int y_start = y - padding_;

			// 预计算源图有效读取区域 (避免循环内重复计算)
			int src_x = (x_start < 0) ? 0 : x_start;
			int src_y = (y_start < 0) ? 0 : y_start;
			int src_end_x =
				(x_start + patch_size_ > fullW) ? fullW : x_start + patch_size_;
			int src_end_y =
				(y_start + patch_size_ > fullH) ? fullH : y_start + patch_size_;

			int src_w = src_end_x - src_x;
			int src_h = src_end_y - src_y;

			// 目标 Patch 偏移
			int dst_x = src_x - x_start;
			int dst_y = src_y - y_start;

			// 补边参数
			int top = dst_y;
			int bottom = patch_size_ - (dst_y + src_h);
			int left = dst_x;
			int right = patch_size_ - (dst_x + src_w);

			// --- B. 构建 Mosaic (OpenMP 并行加速) ---
			if (src_w > 0 && src_h > 0) {
				cv::Rect srcRectValid(src_x, src_y, src_w, src_h);

				int totalViews = ang_res_ * ang_res_;

// 【关键修复】手动展平循环，兼容 MSVC，消除 C4849 警告
#pragma omp parallel for
				for (int i = 0; i < totalViews; ++i) {
					// 反解 u, v
					int u = i / ang_res_;
					int v = i % ang_res_;
					int viewIdx = i;

					// 1. 获取 Wrapper 中的子区域 (指向 inputBuffer 的一部分)
					cv::Rect subMosaicRoi(v * patch_size_, u * patch_size_,
										  patch_size_, patch_size_);
					cv::Mat subMosaicMat = mosaicWrapper(subMosaicRoi);

					// 2. 拷贝 + 补边 (直接写入 inputBuffer)
					cv::copyMakeBorder(grayViews[viewIdx](srcRectValid),
									   subMosaicMat, top, bottom, left, right,
									   cv::BORDER_REPLICATE);
				}
			} else {
				std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
			}

			// --- C. TRT 推理 ---
			// 直接传 vector.data()，绝对安全
			net_->Infer(inputBuffer.data(), outputBuffer.data());

			// --- D. 结果拼接 ---
			// 计算有效写回区域
			int valid_w = (x + step > fullW) ? (fullW - x) : step;
			int valid_h = (y + step > fullH) ? (fullH - y) : step;

			if (valid_w <= 0 || valid_h <= 0)
				continue;

			cv::Rect validRectGlobal(x, y, valid_w, valid_h);

			// 计算在 Patch 中的相对位置 (交集法)
			cv::Rect patchRectGlobal(x_start, y_start, patch_size_,
									 patch_size_);
			cv::Rect inter = validRectGlobal & patchRectGlobal;

			cv::Rect srcRoi(inter.x - patchRectGlobal.x,
							inter.y - patchRectGlobal.y, inter.width,
							inter.height);

			// 拷贝
			cv::Mat outPatch(patch_size_, patch_size_, CV_32FC1,
							 outputBuffer.data());
			outPatch(srcRoi).copyTo(fullDisp(inter));
		}
	}

	return fullDisp;
}