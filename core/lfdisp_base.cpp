#include "lfdisp_base.h"

#include <iostream>

cv::Mat LFDispBase::run(const std::vector<cv::Mat> &inputViews) {
	// 0. 基础校验
	if (!isEngineLoaded()) {
		std::cerr << "[Error] Engine not loaded!" << std::endl;
		return {};
	}

	size_t inputSize = inputViews.size();
	size_t requiredSize = ang_res_ * ang_res_;

	// 定义一个指针，指向最终要处理的数据（避免不必要的拷贝）
	const std::vector<cv::Mat> *pValidViews = nullptr;

	// 定义一个临时容器，仅在需要裁切时存储数据
	std::vector<cv::Mat> croppedViews;

	// 1. 根据视角数量分情况处理
	if (inputSize < requiredSize) {
		// --- 情况 A: 小于 -> 报错 ---
		std::cerr << "[Error] Input views (" << inputSize
				  << ") are insufficient for model requirement ("
				  << requiredSize << ")." << std::endl;
		return {};
	} else if (inputSize == requiredSize) {
		// --- 情况 B: 等于 -> 直接使用 ---
		// 指针直接指向输入，无任何内存复制
		pValidViews = &inputViews;
	} else {
		// --- 情况 C: 大于 -> 裁切中心 ---
		// 简单的平方数校验
		int inputAng = static_cast<int>(std::round(std::sqrt(inputSize)));
		if (inputAng * inputAng != inputSize) {
			std::cerr << "[Error] Input views size (" << inputSize
					  << ") is not a square number." << std::endl;
			return {};
		}

		// 调用基类的筛选函数获取中心视角
		croppedViews = SelectCentralViews(inputViews);
		if (croppedViews.empty()) {
			return {};
		}

		// 指针指向裁切后的临时容器
		pValidViews = &croppedViews;
	}

	// 2. 预处理：转灰度 + 归一化
	// 使用指针解引用 (*pValidViews) 传入
	std::vector<cv::Mat> grayViews;
	pre_process(*pValidViews, grayViews);

	// 3. 核心推理
	return infer(grayViews);
}

void LFDispBase::pre_process(const std::vector<cv::Mat> &inputBGR,
							 std::vector<cv::Mat> &outGray) {
	outGray.resize(inputBGR.size());

// OpenMP 加速预处理
#pragma omp parallel for
	for (int i = 0; i < (int)inputBGR.size(); ++i) {
		cv::Mat gray, grayFloat;
		if (inputBGR[i].channels() == 3) {
			cv::cvtColor(inputBGR[i], gray, cv::COLOR_BGR2GRAY);
		} else {
			gray = inputBGR[i];
		}
		// 统一转为 Float (0.0 - 1.0)
		gray.convertTo(grayFloat, CV_32F, 1.0f / 255.0f);
		outGray[i] = grayFloat;
	}
}