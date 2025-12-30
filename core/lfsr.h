#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "distgssr.h"
#include "lfparams.h"

#include <filesystem> // C++17
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn_superres.hpp>
#include <string>
#include <vector>

class LFSuperRes {
public:
	explicit LFSuperRes();
	~LFSuperRes() = default;

	// Getters
	LFParamsSR::Type type() const { return _type; }
	int scale() const { return _scale; }

	// Setters - 这些函数现在会设置 Dirty 标志
	void setType(LFParamsSR::Type value);
	void setScale(int value);
	void setPatchSize(int value);
	void setModelPaths(const std::string &opencvPath,
					   const std::string &trtPath);

	// 核心处理接口
	// 单图处理：用于插值或 OpenCV DNN
	cv::Mat upsample(const cv::Mat &src);

	// 多图处理：用于 DistgSSR
	std::vector<cv::Mat> upsample(const std::vector<cv::Mat> &views);

private:
	// 状态检查与加载
	void ensureModelLoaded();
	void loadModel();
	void loadDistgSSR();
	void loadOpenCVDNN();

	// 辅助工具
	std::string getModelNameFromType(LFParamsSR::Type type) const;
	int SRTypeToInterFlag(LFParamsSR::Type type) const;

private:
	// 参数配置
	int _scale = 2;
	LFParamsSR::Type _type = LFParamsSR::Type::CUBIC;
	int _patch_size = 196;
	int _ang_res = 5; // 将硬编码的 5 提取出来

	// 状态管理
	bool _isDirty = true; // 替代字符串比较，性能更好

	// 路径管理
	std::filesystem::path _modelPath;
	std::filesystem::path _trtEnginePath;

	// 推理引擎
	// 注意：建议 DistgSSR 内部实现资源释放接口
	DistgSSR distg;
	cv::dnn_superres::DnnSuperResImpl opencv_dnn_sr;
};

#endif // LFSUPERRES_H