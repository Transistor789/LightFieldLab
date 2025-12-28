#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "distgssr.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn_superres.hpp>
#include <string>
#include <vector>

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
	explicit LFSuperRes();

	ModelType type() const { return _type; }
	double scale() const { return _scale; }

	void setType(ModelType value);
	void setScale(int value);
	void setPatchSize(int value);

	// 确保模型已加载
	void ensureModelLoaded();

	cv::Mat upsample(const cv::Mat &src); // 单图处理：插值、opencv_dnn
	std::vector<cv::Mat> upsample(
		const std::vector<cv::Mat> &views); // 多图处理 DistgSSR

private:
	int _scale = 2; // 默认为 2
	ModelType _type = ModelType::CUBIC;
	int _patch_size = 196; // [新增] DistgSSR 默认 patch

	std::string _current_model_id; // 用于检查是否需要重新加载

	// 两种引擎并存
	DistgSSR distg;
	std::vector<cv::Mat> distg_result;
	cv::dnn_superres::DnnSuperResImpl opencv_dnn_sr;

	std::string _modelPath = "data/opencv_srmodel/";
	std::string _trtEnginePath = "data/"; // [新增] TensorRT 引擎路径

	// 内部函数
	std::string getModelNameFromType(ModelType type);
	cv::Mat upsampleCore(const cv::Mat &src);
	void loadModel();
	int modelTypeToInterFlag(ModelType type) const;
};

#endif