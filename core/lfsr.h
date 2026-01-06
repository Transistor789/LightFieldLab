#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "distgssr.h"
// [移除] #include "lfparams.h"

#include <filesystem>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


class LFSuperRes {
public:
	// [新增] 内部定义枚举，自包含
	enum class Method {
		NEAREST,
		LINEAR,
		CUBIC,
		LANCZOS,
		ESPCN,
		FSRCNN,
		DISTGSSR
	};

	explicit LFSuperRes();
	~LFSuperRes() = default;

	/**
	 * @brief 执行超分辨率处理 (核心接口)
	 * @param views 输入图像列表 (如果是单图算法，会遍历处理;
	 * 如果是DistgSSR，整体处理)
	 * @param method 选择的算法
	 * @return 处理后的图像列表
	 */
	std::vector<cv::Mat> upsample(const std::vector<cv::Mat> &views,
								  Method method);

	/**
	 * @brief 单图处理重载 (方便调用)
	 */
	cv::Mat upsample(const cv::Mat &src, Method method);

	// --- 参数设置 (全局配置) ---
	// Scale, AngRes, PatchSize 不再作为 upsample
	// 的参数传入，而是作为上下文状态设置
	void setScale(int value) { m_targetScale = value; }
	void setPatchSize(int value) { m_targetPatchSize = value; }
	void setAngRes(int value) { m_targetAngRes = value; }

	// 路径设置
	void setModelPaths(const std::string &opencvPath,
					   const std::string &trtPath);

	// Getters
	int getScale() const { return m_targetScale; }

private:
	// 智能检查与加载
	bool checkAndLoadModel(Method targetMethod);

	// 具体加载逻辑
	bool loadDistgSSR();
	bool loadOpenCVDNN(Method method);

	// 辅助工具
	std::string getModelNameFromMethod(Method method) const;
	int MethodToInterFlag(Method method) const;
	bool isDeepLearningMethod(Method method) const;

private:
	// --- 核心算法引擎 ---
	DistgSSR distg;
	cv::dnn_superres::DnnSuperResImpl opencv_dnn_sr;

	// --- 路径管理 ---
	std::filesystem::path m_modelPath;	   // OpenCV 模型目录
	std::filesystem::path m_trtEnginePath; // TensorRT 引擎目录

	// --- 目标参数 (用户设定) ---
	int m_targetScale = 2;
	int m_targetPatchSize = 196;
	int m_targetAngRes = 5;

	// --- 当前已加载/生效的状态 (缓存) ---
	Method m_loadedMethod = Method::CUBIC; // 默认为非DL方法
	int m_loadedScale = -1;
	int m_loadedPatchSize = -1;
	int m_loadedAngRes = -1;
};

#endif // LFSUPERRES_H