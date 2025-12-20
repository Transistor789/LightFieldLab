#include "lfmbase.h"

#include <iostream>

void LFMBase::readEngine(const std::string &enginePath) {
	printf("[LFMBase] Loading Engine: %s ...\n", enginePath.c_str());
	net_ = std::make_unique<TRTWrapper>(enginePath);
	printf("[LFMBase] Engine loaded.\n");
}

std::vector<cv::Mat> LFMBase::SelectCentralViews(
	const std::vector<cv::Mat> &inputViews) {
	size_t totalViews = inputViews.size();
	int inputAngRes = static_cast<int>(std::sqrt(totalViews));

	// 1. 校验正方形
	if (inputAngRes * inputAngRes != totalViews) {
		std::cerr << "[Error] Input views (" << totalViews << ") not square."
				  << std::endl;
		return {};
	}
	// 2. 校验尺寸
	if (inputAngRes < ang_res_) {
		std::cerr << "[Error] Input (" << inputAngRes << ") < Model ("
				  << ang_res_ << ")." << std::endl;
		return {};
	}
	// 3. 直接返回
	if (inputAngRes == ang_res_)
		return inputViews;

	// 4. 裁剪中心
	std::vector<cv::Mat> selectedViews;
	selectedViews.reserve(ang_res_ * ang_res_);
	int startOffset = (inputAngRes - ang_res_) / 2;

	for (int u = 0; u < ang_res_; ++u) {
		for (int v = 0; v < ang_res_; ++v) {
			int idx = (startOffset + u) * inputAngRes + (startOffset + v);
			selectedViews.push_back(inputViews[idx]);
		}
	}
	printf("[LFMBase] Auto-cropped input: %dx%d -> %dx%d\n", inputAngRes,
		   inputAngRes, ang_res_, ang_res_);
	return selectedViews;
}