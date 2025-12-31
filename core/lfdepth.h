#ifndef LFDEPTH_
#define LFDEPTH_

#include "distgdisp.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class LFDisp {
public:
	explicit LFDisp() {}

	bool depth(const std::vector<cv::Mat> &views);
	bool hasResult() const { return !m_rawMap.empty(); }
	const cv::Mat &getRawMap() const;
	cv::Mat getGrayVisual() const;
	cv::Mat getJetVisual() const;
	cv::Mat getPlasmaVisual() const;

	void setAngRes(int angRes) { m_targetAngRes = angRes; }
	void setPatchSize(int patchSize) { m_targetPatchSize = patchSize; }

private:
	bool checkAndLoadModel();
	std::string getModelPath(int angRes, int patchSize) const;

private:
	DistgDisp distg_;
	cv::Mat m_rawMap;

	// --- 参数状态管理 ---
	int m_targetAngRes = 9;		 // 用户设定的目标角度分辨率
	int m_targetPatchSize = 196; // 用户设定的目标 Patch 大小

	int m_loadedAngRes = -1; // 当前已加载模型的参数 (-1 表示未加载)
	int m_loadedPatchSize = -1; // 当前已加载模型的参数
};

#endif