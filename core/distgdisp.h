#ifndef DISTGDISP_H
#define DISTGDISP_H

#include "lfdisp_base.h"

class DistgDisp : public LFDispBase {
public:
	explicit DistgDisp() {
		scale_ = 1;		   // 深度估计不改变分辨率
		ang_res_ = 9;	   // 固定 9x9
		patch_size_ = 128; // 默认，需与 Engine 匹配
		padding_ = 8;
		center_only_ = true;
	}

private:
	// 实现基类的纯虚函数
	cv::Mat infer(const std::vector<cv::Mat> &grayViews) override;
};

#endif // DISTGDISP_H