#ifndef LFDEPTH_H
#define LFDEPTH_H

#include "distgdisp.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum class DEMethod { DistgDisp, OACC };
enum class DEColor { Gray, Jet, Plasma };

class LFDepthEstimation {
public:
	explicit LFDepthEstimation() = default;

	/**
	 * @brief 计算深度图
	 * @param views 输入的多视角图像列表
	 * @param method 选择的算法模型 (DistgDisp 或 OACC)
	 * @return 成功返回 true，失败返回 false
	 */
	bool depth(const std::vector<cv::Mat> &views, DEMethod method);

	// 结果获取与可视化
	bool hasResult() const { return !m_rawMap.empty(); }
	const cv::Mat &getRawMap() const;
	cv::Mat getGrayVisual() const;
	cv::Mat getJetVisual() const;
	cv::Mat getPlasmaVisual() const;
	cv::Mat getVisualizedResult(DEColor color) const;
	// 参数设置 (分辨率和PatchSize通常是全局配置，保留Setter)
	void setAngRes(int angRes) { m_targetAngRes = angRes; }
	void setPatchSize(int patchSize) { m_targetPatchSize = patchSize; }

private:
	// 检查状态并在必要时加载模型
	bool checkAndLoadModel(DEMethod targetMethod);
	// 根据参数生成路径
	std::string getModelPath(DEMethod method, int angRes, int patchSize) const;

private:
	DistgDisp distg_;
	cv::Mat m_rawMap;

	// --- 参数状态管理 ---
	int m_targetAngRes = 9;		 // 用户设定的目标角度分辨率
	int m_targetPatchSize = 128; // 用户设定的目标 Patch 大小

	// --- 当前已加载的模型状态 (缓存) ---
	int m_loadedAngRes = -1;					   // 当前引擎的角度分辨率
	int m_loadedPatchSize = -1;					   // 当前引擎的 Patch 大小
	DEMethod m_loadedMethod = DEMethod::DistgDisp; // 当前引擎的算法类型
};

#endif // LFDEPTH_