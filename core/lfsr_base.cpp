#include "lfsr_base.h"

std::vector<cv::Mat> LFSRBase::run(const std::vector<cv::Mat> &inputViews) {
	if (!net_) {
		std::cerr << "[Error] Engine not loaded!" << std::endl;
		return {};
	}

	size_t inputSize = inputViews.size();
	size_t requiredSize = ang_res_ * ang_res_;
	const std::vector<cv::Mat> *pValidViews = nullptr;
	std::vector<cv::Mat> croppedViews;

	// 1. 筛选与裁切逻辑 (保持不变)
	if (inputSize < requiredSize) {
		std::cerr << "[Error] Input views (" << inputSize
				  << ") < Model requirement (" << requiredSize << ")."
				  << std::endl;
		return {};
	} else if (inputSize == requiredSize) {
		pValidViews = &inputViews;
	} else {
		int inputAng = static_cast<int>(std::round(std::sqrt(inputSize)));
		if (inputAng * inputAng != inputSize) {
			std::cerr << "[Error] Input views size (" << inputSize
					  << ") is not a square number." << std::endl;
			return {};
		}
		croppedViews = SelectCentralViews(inputViews);
		if (croppedViews.empty())
			return {};
		pValidViews = &croppedViews;
	}

	std::vector<cv::Mat> yFloatViews, cbViews, crViews;
	pre_process(*pValidViews, yFloatViews, cbViews, crViews);

	// 3. 推理 (模型只接收 yFloatViews)
	std::vector<cv::Mat> srYList = infer(yFloatViews);
	if (srYList.empty())
		return {};

	std::vector<cv::Mat> srBGRList;
	if (center_only_) {
		int centerIdx = cbViews.empty() ? 0 : (int)cbViews.size() / 2;
		cv::Mat cb = cbViews.empty() ? cv::Mat() : cbViews[centerIdx];
		cv::Mat cr = crViews.empty() ? cv::Mat() : crViews[centerIdx];
		srBGRList.push_back(post_process(srYList[0], cb, cr));
	} else {
		srBGRList.resize(srYList.size());
#pragma omp parallel for
		for (int i = 0; i < (int)srYList.size(); ++i) {
			cv::Mat cb = cbViews.empty() ? cv::Mat() : cbViews[i];
			cv::Mat cr = crViews.empty() ? cv::Mat() : crViews[i];
			srBGRList[i] = post_process(srYList[i], cb, cr);
		}
	}
	return srBGRList;
}

std::vector<cv::Mat> LFSRBase::infer(const std::vector<cv::Mat> &yViews) {
	return {};
}

void LFSRBase::pre_process(const std::vector<cv::Mat> &inputViews,
						   std::vector<cv::Mat> &outYFloat,
						   std::vector<cv::Mat> &outCb,
						   std::vector<cv::Mat> &outCr) {
	size_t numViews = inputViews.size();
	outYFloat.resize(numViews);

	// 获取通道数
	int channels = inputViews[0].channels();
	if (channels == 3) {
		outCb.resize(numViews);
		outCr.resize(numViews);
	}

	int targetH = inputViews[0].rows * scale_;
	int targetW = inputViews[0].cols * scale_;

#pragma omp parallel for
	for (int i = 0; i < (int)numViews; ++i) {
		if (channels == 3) {
			// 3通道：BGR -> YCrCb -> 分离 -> 归一化Y + 缩放CbCr
			cv::Mat ycrcb;
			cv::cvtColor(inputViews[i], ycrcb, cv::COLOR_BGR2YCrCb);
			std::vector<cv::Mat> chs;
			cv::split(ycrcb, chs);
			chs[0].convertTo(outYFloat[i], CV_32F, 1.0f / 255.0f);

			cv::Mat cbUp, crUp;
			cv::resize(chs[1], cbUp, cv::Size(targetW, targetH), 0, 0,
					   cv::INTER_CUBIC);
			cv::resize(chs[2], crUp, cv::Size(targetW, targetH), 0, 0,
					   cv::INTER_CUBIC);
			outCb[i] = cbUp;
			outCr[i] = crUp;
		} else {
			// 1通道：仅执行归一化，保持数值操作一致
			inputViews[i].convertTo(outYFloat[i], CV_32F, 1.0f / 255.0f);
		}
	}
}

cv::Mat LFSRBase::post_process(const cv::Mat &srY, const cv::Mat &cb,
							   const cv::Mat &cr) {
	cv::Mat y8u;
	// 1. 统一执行数值截断，防止溢出变白
	if (srY.depth() == CV_32F) {
		cv::Mat yClipped;
		cv::max(0.0f, cv::min(1.0f, srY), yClipped);
		yClipped.convertTo(y8u, CV_8U, 255.0);
	} else {
		y8u = srY;
	}

	// 2. 如果没有色度通道，直接返回截断后的灰度图
	if (cb.empty() || cr.empty()) {
		return y8u;
	}

	// 3. 3通道：合并并转回 BGR
	std::vector<cv::Mat> channels = {y8u, cb, cr};
	cv::Mat ycrcb, bgr;
	cv::merge(channels, ycrcb);
	cv::cvtColor(ycrcb, bgr, cv::COLOR_YCrCb2BGR);
	return bgr;
}