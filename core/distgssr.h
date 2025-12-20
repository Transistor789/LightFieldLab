#ifndef DISTGSSR_H
#define DISTGSSR_H

#include "lfsr_base.h"

#include <opencv2/opencv.hpp>
#include <vector>

class DistgSSR : public LFSRBase {
public:
	explicit DistgSSR() {
		scale_ = 2;
		ang_res_ = 5;
		patch_size_ = 196;
		padding_ = 8;
		center_only_ = true;
	}

private:
	// 核心推理 (支持 Center/All 模式)
	std::vector<cv::Mat> infer(const std::vector<cv::Mat> &yViews) override;
};

#endif // DISTGSSR_H