#ifndef LFREFOCUS_H
#define LFREFOCUS_H

#include "lfdata.h"

#include <vector>

class LFRefocus {
public:
	LFRefocus() = default;
	~LFRefocus() = default;

	void init(const std::shared_ptr<LFData> &ptr);
	void setLF(const std::shared_ptr<LFData> &ptr);

	// 原有的单张重聚焦接口 (用于实时预览)
	cv::Mat refocusByAlpha(float alpha, int crop = 0);
	// 基础重聚焦单元 (内部使用)
	cv::Mat refocusByShift(float shift, int crop = 0);

	/**
	 * @brief [新增] 全焦图像生成接口 (All-in-Focus)
	 * 内部流程：生成一组堆栈 -> 计算清晰度 -> 像素级融合 -> 返回单张结果
	 * * @param min_shift 起始视差
	 * @param max_shift 结束视差
	 * @param step      步长
	 * @param crop      裁剪
	 * @return cv::Mat  最终融合好的一张全焦图像
	 */
	cv::Mat generateAllInFocus(float min_shift, float max_shift, float step,
							   int crop = 0);

private:
	// 堆栈融合算法 (内部使用)
	// 输入一摞图，输出一张融合图
	cv::Mat mergeFocalStack(const std::vector<cv::Mat> &stack);

private:
	std::shared_ptr<LFData> lf;
	int _views, _len, _center, _type;
	cv::Size _size;
	cv::Mat _xmap, _ymap;
};

#endif