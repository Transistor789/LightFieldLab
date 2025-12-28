#ifndef DISTGDISP_H
#define DISTGDISP_H

#include "lfdisp_base.h"

class DistgDisp : public LFDispBase {
public:
	explicit DistgDisp() {
		ang_res_ = 9;	   // 固定 9x9
		patch_size_ = 196; // 默认，需与 Engine 匹配
		padding_ = 8;
	}

private:
	// 实现基类的纯虚函数
	cv::Mat infer(const std::vector<cv::Mat> &grayViews) override;
};

#endif // DISTGDISP_H