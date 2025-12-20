#ifndef LFMBASE_H
#define LFMBASE_H

#include "trtwrapper.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

class LFMBase {
public:
	LFMBase() = default;
	virtual ~LFMBase() = default;

	// --- 统一的运行接口 (纯虚函数) ---
	// 输入: N张图, 输出: M张图
	// virtual std::vector<cv::Mat> run(
	// 	const std::vector<cv::Mat> &inputViews) = 0;

	// --- 通用资源管理 ---
	bool isEngineLoaded() const { return net_ != nullptr; }

	void readEngine(const std::string &enginePath);

	// --- 通用参数设置 ---
	void setPatchSize(int size) { patch_size_ = size; }
	void setPadding(int padding) { padding_ = padding; }
	void setAngRes(int angRes) { ang_res_ = angRes; }
	void setCenterOnly(bool value) { center_only_ = value; }
	void setScale(int value) { scale_ = value; }

protected:
	// 核心组件
	std::unique_ptr<TRTWrapper> net_;

	// 通用配置
	int scale_ = 2;
	int patch_size_ = 128;
	int padding_ = 8;
	int ang_res_ = 5;
	bool center_only_ = false;

	// --- 通用工具：中心视点筛选 ---
	std::vector<cv::Mat> SelectCentralViews(
		const std::vector<cv::Mat> &inputViews);
};

#endif