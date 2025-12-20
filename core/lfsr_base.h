#ifndef LFSRBASE_H
#define LFSRBASE_H

#include "lfmbase.h"

class LFSRBase : public LFMBase {
public:
	LFSRBase() = default;
	virtual ~LFSRBase() = default;

	// --- 统一的运行接口 (纯虚函数) ---
	// 输入: N张图, 输出: M张图
	std::vector<cv::Mat> run(const std::vector<cv::Mat> &inputViews);

protected:
	virtual std::vector<cv::Mat> infer(const std::vector<cv::Mat> &yViews);

	void pre_process(const std::vector<cv::Mat> &inputBGR,
					 std::vector<cv::Mat> &outYFloat,
					 std::vector<cv::Mat> &outCb, std::vector<cv::Mat> &outCr);

	cv::Mat post_process(const cv::Mat &srY, const cv::Mat &cb,
						 const cv::Mat &cr);
};

#endif