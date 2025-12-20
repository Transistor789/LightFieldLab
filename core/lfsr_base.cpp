#include "lfsr_base.h"

std::vector<cv::Mat> LFSRBase::run(const std::vector<cv::Mat> &inputViews) {
	if (!net_) {
		std::cerr << "[Error] Engine not loaded!" << std::endl;
		return {};
	}

	// 1. 筛选
	std::vector<cv::Mat> validViews = SelectCentralViews(inputViews);
	if (validViews.empty())
		return {};

	int H = validViews[0].rows;
	int W = validViews[0].cols;

	// printf("[%s] Run... Input: %dx%d, Scale: %d, Mode: %s\n",
	// 	   getModelName().c_str(), H, W, scale_,
	// 	   center_only_ ? "Center" : "All");

	// 2. 预处理 (通用)
	std::vector<cv::Mat> yFloatViews, cbViews, crViews;
	pre_process(validViews, yFloatViews, cbViews, crViews);

	// 3. 推理 (多态调用子类实现)
	std::vector<cv::Mat> srYList = infer(yFloatViews);
	if (srYList.empty())
		return {};

	// 4. 后处理 (通用)
	std::vector<cv::Mat> srBGRList;

	if (center_only_) {
		// Mode A: Center Only
		// infer 在 center_only 为 true 时只返回 1 张 Y
		int centerIdx = cbViews.size() / 2;
		cv::Mat bgr =
			post_process(srYList[0], cbViews[centerIdx], crViews[centerIdx]);
		srBGRList.push_back(bgr);
	} else {
		// Mode B: All Views
		if (srYList.size() != cbViews.size()) {
			std::cerr << "[Error] Size mismatch in post_process." << std::endl;
			return {};
		}
		srBGRList.resize(srYList.size());

#pragma omp parallel for
		for (int i = 0; i < (int)srYList.size(); ++i) {
			srBGRList[i] = post_process(srYList[i], cbViews[i], crViews[i]);
		}
	}

	return srBGRList;
}

std::vector<cv::Mat> LFSRBase::infer(const std::vector<cv::Mat> &yViews) {
	return {};
}

void LFSRBase::pre_process(const std::vector<cv::Mat> &inputBGR,
						   std::vector<cv::Mat> &outYFloat,
						   std::vector<cv::Mat> &outCb,
						   std::vector<cv::Mat> &outCr) {
	size_t numViews = inputBGR.size();
	outYFloat.resize(numViews);
	outCb.resize(numViews);
	outCr.resize(numViews);

	int targetH = inputBGR[0].rows * scale_;
	int targetW = inputBGR[0].cols * scale_;

#pragma omp parallel for
	for (int i = 0; i < (int)numViews; ++i) {
		cv::Mat ycrcb;
		cv::cvtColor(inputBGR[i], ycrcb, cv::COLOR_BGR2YCrCb);

		std::vector<cv::Mat> channels;
		cv::split(ycrcb, channels);

		// Y: Normalize
		cv::Mat yf;
		channels[0].convertTo(yf, CV_32F, 1.0f / 255.0f);
		outYFloat[i] = yf;

		// Cb/Cr: Resize
		cv::Mat cbUp, crUp;
		cv::resize(channels[1], cbUp, cv::Size(targetW, targetH), 0, 0,
				   cv::INTER_CUBIC);
		cv::resize(channels[2], crUp, cv::Size(targetW, targetH), 0, 0,
				   cv::INTER_CUBIC);

		outCb[i] = cbUp;
		outCr[i] = crUp;
	}
}

cv::Mat LFSRBase::post_process(const cv::Mat &srY, const cv::Mat &cb,
							   const cv::Mat &cr) {
	cv::Mat y8u;
	if (srY.depth() == CV_32F) {
		cv::Mat yClipped;
		cv::max(0.0f, cv::min(1.0f, srY), yClipped);
		yClipped.convertTo(y8u, CV_8U, 255.0);
	} else {
		y8u = srY;
	}
	std::vector<cv::Mat> channels = {y8u, cb, cr};
	cv::Mat ycrcb, bgr;
	cv::merge(channels, ycrcb);
	cv::cvtColor(ycrcb, bgr, cv::COLOR_YCrCb2BGR);
	return bgr;
}