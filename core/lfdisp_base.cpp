#include "lfdisp_base.h"

#include <iostream>

cv::Mat LFDispBase::run(const std::vector<cv::Mat> &inputViews) {
	// 0. 基础校验
	if (!isEngineLoaded()) {
		std::cerr << "[Error] Engine not loaded!" << std::endl;
		return {};
	}

	// 1. 筛选视角 (关键修复)
	// 这一步会根据 ang_res_ (例如 9) 检查 inputViews 是否合法
	// 如果 inputViews 是 13x13 而模型只需要 9x9，这里会自动裁剪中心
	std::vector<cv::Mat> validViews = SelectCentralViews(inputViews);

	// 如果筛选失败（尺寸不对或非正方形），直接返回
	if (validViews.empty()) {
		return {};
	}

	// 2. 预处理：转灰度 + 归一化
	// 注意：这里必须传入 validViews，而不是原始的 inputViews
	std::vector<cv::Mat> grayViews;
	pre_process(validViews, grayViews);

	// 3. 核心推理 (多态调用子类 DistgDisp 的实现)
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