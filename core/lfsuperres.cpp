#include "lfsuperres.h"

#include <stdexcept> // for exceptions

LFSuperRes::LFSuperRes()
	: _scale(2.0), _type(ModelType::CUBIC), _current_model_file("") {}
LFSuperRes::LFSuperRes(std::string filename)
	: _scale(2.0), _type(ModelType::CUBIC), _current_model_file("") {}

int LFSuperRes::modelTypeToInterFlag(ModelType type) const {
	switch (type) {
		case ModelType::NEAREST:
			return cv::INTER_NEAREST;
		case ModelType::BILINEAR:
			return cv::INTER_LINEAR;
		case ModelType::CUBIC:
			return cv::INTER_CUBIC;
		case ModelType::LANCZOS:
			return cv::INTER_LANCZOS4;
		default:
			return -1; // DL模型不需要这个
	}
}

std::string LFSuperRes::getModelNameFromType(ModelType type) {
	switch (type) {
		case ModelType::EDSR:
			return "edsr";
		case ModelType::ESPCN:
			return "espcn";
		case ModelType::FSRCNN:
			return "fsrcnn";
		default:
			throw std::invalid_argument("Invalid ModelType for DL loading.");
	}
}

void LFSuperRes::setType(ModelType value) { _type = value; }

void LFSuperRes::setScale(int value) {
	_scale = 2.0 + static_cast<float>(value);
}

// 模型加载辅助函数：实际的加载和设置逻辑
void LFSuperRes::loadModel() {
	int scale_int = static_cast<int>(_scale);
	std::string name;

	try {
		name = getModelNameFromType(_type);
	} catch (const std::invalid_argument &e) {
		qWarning() << "Error in loadModel: " << e.what();
		return;
	}

	std::string suffix = "_x" + std::to_string(scale_int) + ".pb";
	std::string model_file_path = _modelPath + name + suffix;

	_sr.readModel(model_file_path);
	_sr.setModel(name, scale_int);

	_current_model_file = model_file_path;
}

void LFSuperRes::ensureModelLoaded() {
	if (static_cast<int>(_type) < static_cast<int>(ModelType::EDSR)) {
		// 如果是传统插值，无需加载模型
		_current_model_file = ""; // 清空，确保后续切换DL时能加载
		return;
	}

	int scale_int = static_cast<int>(_scale);
	std::string name;
	try {
		name = getModelNameFromType(_type);
	} catch (...) {
		// 处理错误的DL类型
		return;
	}

	std::string suffix = "_x" + std::to_string(scale_int) + ".pb";
	std::string new_model_file = _modelPath + name + suffix;

	// 检查是否已经加载
	if (new_model_file == _current_model_file) {
		return;
	}

	// 必须加载新模型
	loadModel();
}

// 核心超分逻辑：统一处理插值和DL模型
cv::Mat LFSuperRes::upsampleCore(const cv::Mat &src) {
	cv::Mat dst;
	if (static_cast<int>(_type) < static_cast<int>(ModelType::EDSR)) {
		// 传统插值方法
		int type_flag = modelTypeToInterFlag(_type);
		cv::resize(src, dst, cv::Size(), _scale, _scale, type_flag);

		// 确保转换后的类型匹配 LightField 数据的位深，这里使用 lf->channels
		// 是正确的
		if (lf) {
			dst.convertTo(dst, CV_8UC(lf->channels));
		} else {
			// 如果 lf 为空，使用默认 8U3C
			dst.convertTo(dst, CV_8UC3);
		}

	} else if (static_cast<int>(_type) < 7) {
		// DL 模型方法
		// 必须确保模型已加载
		ensureModelLoaded();
		_sr.upsample(src, dst);

	} else {
		std::cout << "Unsupported Super Resolution type: "
				  << static_cast<int>(_type);
		return cv::Mat{};
	}

	return dst;
}

// 公共接口：单图像超分
cv::Mat LFSuperRes::upsample(const cv::Mat &src) { return upsampleCore(src); }

cv::Mat LFSuperRes::upsample_multiple() {
	// 实际实现应在这里遍历所有视点并调用 upsampleCore 或 upsample_single
	// ...
	return cv::Mat{};
}