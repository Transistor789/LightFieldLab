#include "lfsr.h"

#include <format> // C++20
#include <iostream>
#include <opencv2/highgui.hpp>

LFSuperResolution::LFSuperResolution() {
	m_modelPath = "models/";
	m_trtEnginePath = "models/";
}

void LFSuperResolution::setModelPaths(const std::string &opencvPath,
									  const std::string &trtPath) {
	m_modelPath = opencvPath;
	m_trtEnginePath = trtPath;
	// 路径改变强制重置缓存状态，确保下次重新加载
	m_loadedScale = -1;
}

// --- 核心多图接口 ---
std::vector<cv::Mat> LFSuperResolution::upsample(
	const std::vector<cv::Mat> &views, SRMethod method) {
	if (views.empty()) {
		std::cerr << "[LFSuperResolution] Error: No views loaded!" << std::endl;
		return {};
	}

	// [新增] 强制检查 8-bit
	if (views[0].depth() != CV_8U) {
		std::cerr << "[LFSuperResolution] Error: Input must be 8-bit (CV_8U). "
					 "Current "
					 "depth: "
				  << views[0].depth() << std::endl;
		return {};
	}

	// 1. 智能加载/切换模型
	if (!checkAndLoadModel(method)) {
		std::cerr
			<< "[LFSuperResolution] Model load failed. Falling back to LINEAR."
			<< std::endl;
		if (method != SRMethod::LINEAR) {
			return upsample(views, SRMethod::LINEAR);
		}
		return {};
	}

	// 2. 根据方法分发执行
	if (method == SRMethod::DISTGSSR) {
		return distg.run(views);
	} else {
		std::vector<cv::Mat> results;
		results.reserve(views.size());

		if (isDeepLearningMethod(method)) {
			for (const auto &view : views) {
				cv::Mat dst;
				opencv_dnn_sr.upsample(view, dst);
				results.push_back(dst);
			}
		} else {
			int flag = MethodToInterFlag(method);
			for (const auto &view : views) {
				cv::Mat dst;
				cv::resize(view, dst, cv::Size(), m_targetScale, m_targetScale,
						   flag);
				results.push_back(dst);
			}
		}
		return results;
	}
}

// --- 单图接口重载 ---
cv::Mat LFSuperResolution::upsample(const cv::Mat &src, SRMethod method) {
	if (src.empty())
		return cv::Mat();

	// [新增] 强制检查 8-bit
	if (src.depth() != CV_8U) {
		std::cerr << "[LFSuperResolution] Error: Input image must be 8-bit."
				  << std::endl;
		return cv::Mat();
	}

	std::vector<cv::Mat> inputs = {src};

	if (method == SRMethod::DISTGSSR) {
		std::cerr << "[LFSuperResolution] Warning: DistgSSR requires Light "
					 "Field (vector)."
				  << std::endl;
	}

	std::vector<cv::Mat> results = upsample(inputs, method);

	if (results.empty())
		return cv::Mat();
	return results[0];
}

// --- 智能加载逻辑 ---
bool LFSuperResolution::checkAndLoadModel(SRMethod targetMethod) {
	// 1. 检查是否是传统插值方法 (无需加载文件)
	if (!isDeepLearningMethod(targetMethod)) {
		// 仅更新状态记录，直接返回成功
		m_loadedMethod = targetMethod;
		m_loadedScale = m_targetScale;
		return true;
	}

	// 2. 检查状态变更
	bool methodChanged = (m_loadedMethod != targetMethod);
	bool scaleChanged = (m_loadedScale != m_targetScale);

	// DistgSSR 特有的检查参数
	bool distgParamChanged = false;
	if (targetMethod == SRMethod::DISTGSSR) {
		distgParamChanged = (m_loadedPatchSize != m_targetPatchSize)
							|| (m_loadedAngRes != m_targetAngRes);
	}

	// 如果一切未变，且确实是DL方法，无需重载
	// (注意：这里假设之前的加载是成功的。为了更严谨，可以加一个 m_isModelValid
	// 标志)
	if (!methodChanged && !scaleChanged && !distgParamChanged) {
		return true;
	}

	std::cout
		<< "[LFSuperResolution] State change detected. Preparing to load..."
		<< std::endl;

	// 3. 执行加载
	bool success = false;
	if (targetMethod == SRMethod::DISTGSSR) {
		success = loadDistgSSR();
	} else {
		success = loadOpenCVDNN(targetMethod);
	}

	// 4. 更新缓存状态
	if (success) {
		m_loadedMethod = targetMethod;
		m_loadedScale = m_targetScale;
		m_loadedPatchSize = m_targetPatchSize;
		m_loadedAngRes = m_targetAngRes;
	} else {
		// 加载失败，重置状态
		m_loadedScale = -1;
	}

	return success;
}

bool LFSuperResolution::loadDistgSSR() {
	int totalRes = m_targetAngRes * m_targetPatchSize;

	// 检查 Scale 支持 (DistgSSR 通常只支持 x2, x4)
	if (m_targetScale != 2 && m_targetScale != 4) {
		std::cerr
			<< "[LFSuperResolution] Error: DistgSSR only supports scale 2 or 4."
			<< std::endl;
		return false;
	}

	// 构造路径
#ifdef _WIN32
	std::string osSuffix = "Windows";
#elif __linux__
	std::string osSuffix = "Linux";
#else
	std::string osSuffix = "Unknown";
#endif

	std::string engineName =
		std::format("DistgSSR_{}x_1x1x{}x{}_FP16_{}.engine", m_targetScale,
					totalRes, totalRes, osSuffix);
	auto fullPath = m_trtEnginePath / engineName;

	std::cout << "[LFSuperResolution] Loading DistgSSR Engine: "
			  << fullPath.string() << std::endl;

	if (!std::filesystem::exists(fullPath)) {
		std::cerr << "[LFSuperResolution] File not found: " << fullPath.string()
				  << std::endl;
		return false;
	}

	try {
		// 同步参数
		distg.setScale(m_targetScale);
		distg.setPatchSize(m_targetPatchSize);
		distg.setAngRes(m_targetAngRes); // 假设 DistgSSR 有此接口

		distg.readEngine(fullPath.string());
		return true;
	} catch (const std::exception &e) {
		std::cerr << "[LFSuperResolution] DistgSSR load exception: " << e.what()
				  << std::endl;
		return false;
	}
}

bool LFSuperResolution::loadOpenCVDNN(SRMethod method) {
	std::string modelName = getModelNameFromMethod(method);
	if (modelName.empty())
		return false;

	// 格式: path/espcn_x2.pb
	// 注意：文件名格式需要根据你实际的数据集调整，比如有些是
	// ESPCN_x2.pb，有些是 espcn_x2.pb
	std::string fileName = std::format("{}_x{}.pb", modelName, m_targetScale);
	auto fullPath = m_modelPath / fileName;

	std::cout << "[LFSuperResolution] Loading OpenCV Model: "
			  << fullPath.string() << std::endl;

	if (!std::filesystem::exists(fullPath)) {
		std::cerr << "[LFSuperResolution] File not found: " << fullPath.string()
				  << std::endl;
		return false;
	}

	try {
		opencv_dnn_sr.readModel(fullPath.string());
		opencv_dnn_sr.setModel(modelName, m_targetScale);
		return true;
	} catch (const cv::Exception &e) {
		std::cerr << "[LFSuperResolution] OpenCV DNN load exception: "
				  << e.what() << std::endl;
		return false;
	}
}

// --- 辅助函数 ---

std::string LFSuperResolution::getModelNameFromMethod(SRMethod method) const {
	switch (method) {
		case SRMethod::ESPCN:
			return "espcn";
		case SRMethod::FSRCNN:
			return "fsrcnn";
		// 如果有 EDSR 等其他模型可以在此添加
		default:
			return "";
	}
}

int LFSuperResolution::MethodToInterFlag(SRMethod method) const {
	switch (method) {
		case SRMethod::NEAREST:
			return cv::INTER_NEAREST;
		case SRMethod::CUBIC:
			return cv::INTER_CUBIC;
		case SRMethod::LANCZOS:
			return cv::INTER_LANCZOS4;
		default:
			return cv::INTER_LINEAR;
	}
}

bool LFSuperResolution::isDeepLearningMethod(SRMethod method) const {
	return (method == SRMethod::ESPCN || method == SRMethod::FSRCNN
			|| method == SRMethod::DISTGSSR);
}