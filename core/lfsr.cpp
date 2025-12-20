#include "lfsr.h"

#include <iostream>

LFSuperRes::LFSuperRes() {
	// 默认初始化 DistgSSR
	// 此时还没有加载 Engine，直到 ensureModelLoaded 被调用
}

void LFSuperRes::setType(ModelType value) {
	if (_type != value) {
		_type = value;
		_current_model_id = ""; // 强制下次 ensureModelLoaded 时重载
	}
}

void LFSuperRes::setScale(int value) {
	if (_scale != value) {
		_scale = value;
		_current_model_id = ""; // 强制重载

		// 同步更新 DistgSSR 的参数
		distg.setScale(value);
	}
}

void LFSuperRes::setPatchSize(int value) {
	if (_patch_size != value) {
		_patch_size = value;
		// DistgSSR 的 Engine 是和 PatchSize 绑定的，改变后必须重载 Engine
		if (_type == ModelType::DISTGSSR) {
			_current_model_id = "";
		}
		distg.setPatchSize(value);
	}
}

void LFSuperRes::ensureModelLoaded() {
	// 生成一个当前配置的唯一 ID
	std::string new_id;
	if (_type == ModelType::DISTGSSR) {
		new_id = std::format("DISTGSSR_x{}_p{}", _scale, _patch_size);
	} else {
		new_id = std::format("OPENCV_t{}_x{}", (int)_type, _scale);
	}

	// 如果配置变了，或者是首次运行
	if (_current_model_id != new_id) {
		loadModel();
		_current_model_id = new_id;
	}
}

void LFSuperRes::loadModel() {
	// ==========================================
	// 分支 1: DistgSSR (TensorRT)
	// ==========================================
	if (_type == ModelType::DISTGSSR) {
		// 构造 Engine 文件名
		// 假设文件名格式: DistgSSR_2x_5x5_1x1x640x640.engine
		int angRes = 5; // 你的模型固定是 5x5
		int totalRes = angRes * _patch_size;

		std::string engineName = std::format(
			"DistgSSR_{}x_5x5_1x1x{}x{}.engine", _scale, totalRes, totalRes);

		std::string fullPath = _trtEnginePath + engineName;

		std::cout << "[LFSuperRes] Loading TRT Engine: " << fullPath
				  << std::endl;

		// 这里的 try-catch 很重要，防止 Engine 不存在导致崩毁
		try {
			distg.readEngine(fullPath);
		} catch (const std::exception &e) {
			std::cerr << "[Error] Failed to load DistgSSR engine: " << e.what()
					  << std::endl;
			// 回退到双线性，防止程序崩溃
			_type = ModelType::LINEAR;
		}
		return;
	}

	// ==========================================
	// 分支 2: OpenCV DNN 模型 (ESPCN, FSRCNN)
	// ==========================================
	if (_type >= ModelType::ESPCN && _type <= ModelType::FSRCNN) {
		std::string modelName = getModelNameFromType(_type); // e.g., "espcn"
		std::string fileName =
			std::format("{}{}_x{}.pb", _modelPath, modelName, _scale);

		std::cout << "[LFSuperRes] Loading OpenCV Model: " << fileName
				  << std::endl;
		try {
			opencv_dnn_sr.readModel(fileName);
			opencv_dnn_sr.setModel(modelName, _scale);
		} catch (const cv::Exception &e) {
			std::cerr << "[Error] OpenCV SR load failed: " << e.what()
					  << std::endl;
			_type = ModelType::LINEAR;
		}
		return;
	}

	// ==========================================
	// 分支 3: 传统插值 (不需要加载模型)
	// ==========================================
	// Do nothing
}

// 单图处理 (DistgSSR 不支持单图模式，这里做个保护)
cv::Mat LFSuperRes::upsample(const cv::Mat &src) {
	ensureModelLoaded();

	if (_type == ModelType::DISTGSSR) {
		std::cerr << "[Warning] DistgSSR requires multiple views! Calling "
					 "upsample instead..."
				  << std::endl;
		// 如果用户错误调用了单图接口，DistgSSR 无法工作，只能降级为双线性
		cv::Mat dst;
		cv::resize(src, dst, cv::Size(), _scale, _scale, cv::INTER_LINEAR);
		return dst;
	}

	return upsampleCore(src);
}

// 核心逻辑：多图处理
cv::Mat LFSuperRes::upsample() {
	ensureModelLoaded();

	if (!lf || lf->data.empty()) {
		std::cerr << "[Error] No Light Field data loaded!" << std::endl;
		return cv::Mat();
	}

	// 1. 准备数据
	// DistgSSR 需要 5x5 的输入。假设 lf->data 是 9x9 (81张)
	// 我们需要提取中心的 5x5
	std::vector<cv::Mat> inputViews;

	int sourceGrid = 9; // 假设你的 LfData 是 9x9
	int modelGrid = 5;
	int start_idx = (sourceGrid - modelGrid) / 2; // (9-5)/2 = 2

	for (int u = 0; u < modelGrid; ++u) {
		for (int v = 0; v < modelGrid; ++v) {
			int src_u = start_idx + u;
			int src_v = start_idx + v;
			int idx = src_u * sourceGrid + src_v;

			// 边界检查
			if (idx < lf->data.size()) {
				inputViews.push_back(lf->data[idx]);
			}
		}
	}

	// 2. 调用 DistgSSR
	// process 返回 vector<Mat>，通常只有一张结果
	std::vector<cv::Mat> results =
		distg.run(inputViews); // 注意：这里需要你把 DistgSSR::Process
							   // 改名为 upsample 或统一接口

	return results[results.size() / 2];
}

cv::Mat LFSuperRes::upsampleCore(const cv::Mat &src) {
	cv::Mat dst;

	// OpenCV DNN
	if (_type >= ModelType::ESPCN && _type <= ModelType::FSRCNN) {
		opencv_dnn_sr.upsample(src, dst);
	}
	// 传统插值
	else {
		int flag = modelTypeToInterFlag(_type);
		cv::resize(src, dst, cv::Size(), _scale, _scale, flag);
	}
	return dst;
}

std::string LFSuperRes::getModelNameFromType(ModelType type) {
	switch (type) {
		case ModelType::ESPCN:
			return "espcn";
		case ModelType::FSRCNN:
			return "fsrcnn";
		default:
			return "";
	}
}

int LFSuperRes::modelTypeToInterFlag(ModelType type) const {
	switch (type) {
		case ModelType::NEAREST:
			return cv::INTER_NEAREST;
		case ModelType::LINEAR:
			return cv::INTER_LINEAR;
		case ModelType::CUBIC:
			return cv::INTER_CUBIC;
		case ModelType::LANCZOS:
			return cv::INTER_LANCZOS4;
		default:
			return cv::INTER_LINEAR;
	}
}