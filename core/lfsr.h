#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "distgssr.h"
#include "lfdata.h"

#include <format> // C++20，如果用旧标准可以使用 sprintf
#include <opencv2/dnn_superres.hpp>
#include <string>

// ModelType 枚举保持不变
enum class ModelType {
	NEAREST = 0,
	LINEAR = 1,
	CUBIC = 2,
	LANCZOS = 3,

	ESPCN = 4,
	FSRCNN = 5,

	DISTGSSR = 6 // 光场专用模型
};

class LFSuperRes {
public:
	LFSuperRes();
	LFSuperRes(std::string filename);

	ModelType type() const { return _type; }
	double scale() const { return _scale; }

	void setType(ModelType value);
	void setScale(int value);

	// [新增] 设置 DistgSSR 专用的 PatchSize
	void setPatchSize(int value);

	void setLF(const LfPtr &ptr) { lf = ptr; }

	// 确保模型已加载
	void ensureModelLoaded();

	// 单图处理 (用于普通插值或 SISR)
	cv::Mat upsample(const cv::Mat &src);

	// [核心] 处理整个光场 (DistgSSR 入口)
	// 对于 SISR，它可能循环处理所有图
	// 对于 DistgSSR，它读取 lf 中的 vector<Mat> 并融合
	cv::Mat upsample();

	LfPtr lf;

private:
	int _scale = 2; // 默认为 2
	ModelType _type = ModelType::CUBIC;
	int _patch_size = 128; // [新增] DistgSSR 默认 patch

	std::string _current_model_id; // 用于检查是否需要重新加载

	// 两种引擎并存
	DistgSSR distg;
	cv::dnn_superres::DnnSuperResImpl opencv_dnn_sr;

	std::string _modelPath = "../data/opencv_srmodel/";
	std::string _trtEnginePath = "../data/"; // [新增] TensorRT 引擎路径

	// 内部函数
	std::string getModelNameFromType(ModelType type);
	cv::Mat upsampleCore(const cv::Mat &src);
	void loadModel();
	int modelTypeToInterFlag(ModelType type) const;
};

#endif