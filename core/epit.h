#ifndef EPIT_H
#define EPIT_H

#include "lfmbase.h"
#include "lfsr_base.h"

#include <opencv2/opencv.hpp>
#include <vector>

class EPIT : public LFSRBase {
public:
	explicit EPIT() {
		scale_ = 2;
		ang_res_ = 5;
		patch_size_ = 64;
		padding_ = 8;
		center_only_ = true;
	}

	// 执行推理
	// std::vector<cv::Mat> run(const std::vector<cv::Mat> &inputViews)
	// override;

private:
	// 预处理: 生成所有视角的 Y(Float) 和 插值放大的 Cb/Cr
	// void pre_process(const std::vector<cv::Mat> &inputBGR,
	// 				std::vector<cv::Mat> &outYFloat,
	// 				std::vector<cv::Mat> &outCb, std::vector<cv::Mat> &outCr);

	// 核心推理: 适配 EPIT 的 Stack 输入布局
	std::vector<cv::Mat> infer(const std::vector<cv::Mat> &yViews) override;

	// 后处理: 智能类型检测与合并
	// cv::Mat post_process(const cv::Mat &srY, const cv::Mat &cb,
	// 					const cv::Mat &cr);
};

#endif // EPIT_H