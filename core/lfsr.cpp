#include "lfsr.h"

#include <format> // C++20
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// 构造函数：初始化路径
LFSuperRes::LFSuperRes() {
	_modelPath = "data/opencv_srmodel/";
	_trtEnginePath = "data/";
}

void LFSuperRes::setModelPaths(const std::string &opencvPath,
							   const std::string &trtPath) {
	_modelPath = opencvPath;
	_trtEnginePath = trtPath;
	// 路径改变可能意味着需要重新加载（尽管通常不需要，但在找不到文件重试时有用）
	_isDirty = true;
}

void LFSuperRes::setType(LFParamsSR::Type value) {
	if (_type != value) {
		_type = value;
		_isDirty = true; // 标记需要重载

		// 优化：切换类型时，尝试释放不用的资源以节省显存
		if (_type != LFParamsSR::Type::DISTGSSR) {
			// 如果 DistgSSR 有 release/clear 方法，在这里调用
			// distg.release();
		}
	}
}

void LFSuperRes::setScale(int value) {
	if (_scale != value) {
		_scale = value;
		_isDirty = true;

		if (value == 2 || value == 4) {
			distg.setScale(value);
		} else {
// 仅在 Debug 模式或 Verbose 模式下打印，避免刷屏
#ifdef _DEBUG
			std::cerr << "[LFSuperRes] Warning: DistgSSR only supports scale 2 "
						 "or 4.\n";
#endif
		}
	}
}

void LFSuperRes::setPatchSize(int value) {
	if (_patch_size != value) {
		_patch_size = value;
		// 只有当使用 DistgSSR 时，PatchSize 改变才需要重载引擎
		if (_type == LFParamsSR::Type::DISTGSSR) {
			_isDirty = true;
		}
		distg.setPatchSize(value);
	}
}

void LFSuperRes::ensureModelLoaded() {
	// 性能优化：只检查布尔标志位
	if (_isDirty) {
		loadModel();
		_isDirty = false;
	}
}

void LFSuperRes::loadModel() {
	// 根据类型分发加载逻辑
	if (_type == LFParamsSR::Type::DISTGSSR) {
		loadDistgSSR();
	} else if (_type >= LFParamsSR::Type::ESPCN
			   && _type <= LFParamsSR::Type::FSRCNN) {
		loadOpenCVDNN();
	}
	// 传统插值模式无需加载模型
}

void LFSuperRes::loadDistgSSR() {
	// 构造 Engine 文件名
	int totalRes = _ang_res * _patch_size;

	// 使用 std::filesystem 处理路径拼接，更安全
	std::string engineName = std::format("DistgSSR_{}x_1x1x{}x{}_FP16.engine",
										 _scale, totalRes, totalRes);
	auto fullPath = _trtEnginePath / engineName;

	std::cout << "[LFSuperRes] Loading TRT Engine: " << fullPath.string()
			  << std::endl;

	if (!std::filesystem::exists(fullPath)) {
		std::cerr << "[Error] Engine file not found: " << fullPath.string()
				  << std::endl;
		_type = LFParamsSR::Type::LINEAR; // 自动降级
		return;
	}

	try {
		distg.readEngine(fullPath.string());
	} catch (const std::exception &e) {
		std::cerr << "[Error] Failed to load DistgSSR engine: " << e.what()
				  << std::endl;
		_type = LFParamsSR::Type::LINEAR;
	}
}

void LFSuperRes::loadOpenCVDNN() {
	std::string modelName = getModelNameFromType(_type);
	std::string fileName =
		std::format("{}{}_x{}.pb", "", modelName,
					_scale); // 注意：这里需要根据实际文件名格式调整

	// 路径拼接
	auto fullPath = _modelPath / fileName;

	std::cout << "[LFSuperRes] Loading OpenCV Model: " << fullPath.string()
			  << std::endl;

	if (!std::filesystem::exists(fullPath)) {
		std::cerr << "[Error] Model file not found: " << fullPath.string()
				  << std::endl;
		_type = LFParamsSR::Type::LINEAR;
		return;
	}

	try {
		// 读取新模型前，OpenCV DNN
		// 内部通常会自动处理旧资源，但显式重置是个好习惯 opencv_dnn_sr =
		// cv::dnn_superres::DnnSuperResImpl();
		opencv_dnn_sr.readModel(fullPath.string());
		opencv_dnn_sr.setModel(modelName, _scale);
	} catch (const cv::Exception &e) {
		std::cerr << "[Error] OpenCV SR load failed: " << e.what() << std::endl;
		_type = LFParamsSR::Type::LINEAR;
	}
}

// 单图处理接口
cv::Mat LFSuperRes::upsample(const cv::Mat &src) {
	if (src.empty())
		return cv::Mat();

	ensureModelLoaded();

	// 逻辑保护：如果是 DistgSSR 但只给了一张图，降级处理
	if (_type == LFParamsSR::Type::DISTGSSR) {
		std::cerr << "[Warning] DistgSSR requires Light Field (vector). "
					 "Falling back to LINEAR."
				  << std::endl;
		cv::Mat dst;
		cv::resize(src, dst, cv::Size(), _scale, _scale, cv::INTER_LINEAR);
		return dst;
	}

	// 执行 OpenCV DNN 或 插值
	cv::Mat dst;
	if (_type >= LFParamsSR::Type::ESPCN && _type <= LFParamsSR::Type::FSRCNN) {
		opencv_dnn_sr.upsample(src, dst);
	} else {
		int flag = SRTypeToInterFlag(_type);
		cv::resize(src, dst, cv::Size(), _scale, _scale, flag);
	}
	return dst;
}

// 多图处理接口
std::vector<cv::Mat> LFSuperRes::upsample(const std::vector<cv::Mat> &views) {
	if (views.empty()) {
		std::cerr << "[Error] No Light Field data loaded!" << std::endl;
		return {};
	}

	ensureModelLoaded();

	if (_type == LFParamsSR::Type::DISTGSSR) {
		return distg.run(views);
	}

	// 如果当前选的是单图模型（ESPCN等）或插值，但用户传入了多图
	// 我们遍历处理每一张图
	std::vector<cv::Mat> results;
	results.reserve(views.size());

	for (const auto &view : views) {
		results.push_back(upsample(view)); // 复用单图逻辑
	}
	return results;
}

std::string LFSuperRes::getModelNameFromType(LFParamsSR::Type type) const {
	switch (type) {
		case LFParamsSR::Type::ESPCN:
			return "espcn";
		case LFParamsSR::Type::FSRCNN:
			return "fsrcnn";
		default:
			return "";
	}
}

int LFSuperRes::SRTypeToInterFlag(LFParamsSR::Type type) const {
	switch (type) {
		case LFParamsSR::Type::NEAREST:
			return cv::INTER_NEAREST;
		case LFParamsSR::Type::CUBIC:
			return cv::INTER_CUBIC;
		case LFParamsSR::Type::LANCZOS:
			return cv::INTER_LANCZOS4;
		default:
			return cv::INTER_LINEAR;
	}
}