#ifndef LFDISP_BASE_H
#define LFDISP_BASE_H

#include "lfmbase.h"

#include <omp.h>
#include <opencv2/opencv.hpp>
#include <vector>

class LFDispBase : public LFMBase {
public:
	LFDispBase() = default;
	virtual ~LFDispBase() = default;

	// --- 实现基类统一接口 ---
	// 输入: 81张图, 输出: 1张视差图 (封装在 vector 中)
	virtual cv::Mat run(const std::vector<cv::Mat> &inputViews);

protected:
	// --- 纯虚函数：核心推理 ---
	// 子类需要实现具体的 滑窗(Sliding Window) + 模型推理 逻辑
	// 输入: 81张灰度图 (Gray Views, CV_32FC1 0-1)
	// 返回: 1张完整的视差图 (Disparity Map, CV_32FC1)
	virtual cv::Mat infer(const std::vector<cv::Mat> &grayViews) = 0;

	// --- 通用工具 ---
	// 深度估计预处理：BGR -> Gray (Float 0-1)
	void pre_process(const std::vector<cv::Mat> &inputBGR,
					 std::vector<cv::Mat> &outGray);
};

#endif // LFDISP_BASE_H