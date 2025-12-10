#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "lfdata.h"

#include <opencv2/dnn_superres.hpp>

// 完善的 ModelType
enum class ModelType {
	NEAREST = 0,
	BILINEAR = 1,
	CUBIC = 2,
	LANCZOS = 3,

	EDSR = 4,
	ESPCN = 5,
	FSRCNN = 6
};

class LFSuperRes {
public:
	LFSuperRes();
	LFSuperRes(std::string filename);

	ModelType type() const { return _type; }
	double scale() const { return _scale; }

	void setType(ModelType value);
	void setScale(int value);
	void update(const LfPtr &ptr) { lf = ptr; }

	void ensureModelLoaded();

	cv::Mat upsample(const cv::Mat &src);
	cv::Mat upsample_multiple();

	LfPtr lf;

private:
	float _scale;
	ModelType _type;
	std::string _current_model_file; // 用于延迟加载检查

	// 移除非必要的 cv::Mat 成员，只在需要时在栈上创建
	// cv::Mat _data;
	// cv::Mat _input, _output;

	std::string _modelPath = "data/opencv_srmodel/";

	cv::dnn_superres::DnnSuperResImpl _sr;

	std::string getModelNameFromType(ModelType type);
	cv::Mat upsampleCore(const cv::Mat &src);		// 核心超分逻辑
	void loadModel();								// 实际的加载和设置
	int modelTypeToInterFlag(ModelType type) const; // 辅助转换函数
};

#endif
