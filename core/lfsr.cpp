#include "lfsr.h"

#include <format> // C++20
#include <iostream>

LFSuperRes::LFSuperRes() {
	m_modelPath = "data/opencv_srmodel/";
	m_trtEnginePath = "data/";
}

void LFSuperRes::setModelPaths(const std::string &opencvPath,
							   const std::string &trtPath) {
	m_modelPath = opencvPath;
	m_trtEnginePath = trtPath;
	// 路径改变强制重置缓存状态，确保下次重新加载
	m_loadedScale = -1;
}

// --- 核心多图接口 ---
std::vector<cv::Mat> LFSuperRes::upsample(const std::vector<cv::Mat> &views,
										  Method method) {
	if (views.empty()) {
		std::cerr << "[LFSuperRes] Error: No views loaded!" << std::endl;
		return {};
	}

	// 1. 智能加载/切换模型
	if (!checkAndLoadModel(method)) {
		std::cerr << "[LFSuperRes] Model load failed. Falling back to LINEAR."
				  << std::endl;
		// 递归调用自己，使用备用方案，防止死循环 (LINEAR 不需要加载)
		if (method != Method::LINEAR) {
			return upsample(views, Method::LINEAR);
		}
		return {};
	}

	// 2. 根据方法分发执行
	if (method == Method::DISTGSSR) {
		// DistgSSR 需要整个光场序列
		return distg.run(views);
	} else {
		// 其他方法（插值 或 OpenCV DNN）都是针对单张图处理的
		// 我们在这里遍历处理
		std::vector<cv::Mat> results;
		results.reserve(views.size());

		if (isDeepLearningMethod(method)) {
			// OpenCV DNN 推理
			for (const auto &view : views) {
				cv::Mat dst;
				opencv_dnn_sr.upsample(view, dst);
				results.push_back(dst);
			}
		} else {
			// 传统插值
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
cv::Mat LFSuperRes::upsample(const cv::Mat &src, Method method) {
	if (src.empty())
		return cv::Mat();

	// 包装成 vector 调用核心逻辑
	std::vector<cv::Mat> inputs = {src};

	// 如果是 DistgSSR 但只有一张图，打印警告
	if (method == Method::DISTGSSR) {
		std::cerr
			<< "[LFSuperRes] Warning: DistgSSR requires Light Field (vector). "
			   "Input is single image."
			<< std::endl;
		// 这里 DistgSSR 内部可能会处理，或者我们降级
	}

	std::vector<cv::Mat> results = upsample(inputs, method);

	if (results.empty())
		return cv::Mat();
	return results[0];
}

// --- 智能加载逻辑 ---
bool LFSuperRes::checkAndLoadModel(Method targetMethod) {
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
	if (targetMethod == Method::DISTGSSR) {
		distgParamChanged = (m_loadedPatchSize != m_targetPatchSize)
							|| (m_loadedAngRes != m_targetAngRes);
	}

	// 如果一切未变，且确实是DL方法，无需重载
	// (注意：这里假设之前的加载是成功的。为了更严谨，可以加一个 m_isModelValid
	// 标志)
	if (!methodChanged && !scaleChanged && !distgParamChanged) {
		return true;
	}

	std::cout << "[LFSuperRes] State change detected. Preparing to load..."
			  << std::endl;

	// 3. 执行加载
	bool success = false;
	if (targetMethod == Method::DISTGSSR) {
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

bool LFSuperRes::loadDistgSSR() {
	int totalRes = m_targetAngRes * m_targetPatchSize;

	// 检查 Scale 支持 (DistgSSR 通常只支持 x2, x4)
	if (m_targetScale != 2 && m_targetScale != 4) {
		std::cerr << "[LFSuperRes] Error: DistgSSR only supports scale 2 or 4."
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

	std::cout << "[LFSuperRes] Loading DistgSSR Engine: " << fullPath.string()
			  << std::endl;

	if (!std::filesystem::exists(fullPath)) {
		std::cerr << "[LFSuperRes] File not found: " << fullPath.string()
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
		std::cerr << "[LFSuperRes] DistgSSR load exception: " << e.what()
				  << std::endl;
		return false;
	}
}

bool LFSuperRes::loadOpenCVDNN(Method method) {
	std::string modelName = getModelNameFromMethod(method);
	if (modelName.empty())
		return false;

	// 格式: path/espcn_x2.pb
	// 注意：文件名格式需要根据你实际的数据集调整，比如有些是
	// ESPCN_x2.pb，有些是 espcn_x2.pb
	std::string fileName = std::format("{}_x{}.pb", modelName, m_targetScale);
	auto fullPath = m_modelPath / fileName;

	std::cout << "[LFSuperRes] Loading OpenCV Model: " << fullPath.string()
			  << std::endl;

	if (!std::filesystem::exists(fullPath)) {
		std::cerr << "[LFSuperRes] File not found: " << fullPath.string()
				  << std::endl;
		return false;
	}

	try {
		opencv_dnn_sr.readModel(fullPath.string());
		opencv_dnn_sr.setModel(modelName, m_targetScale);
		return true;
	} catch (const cv::Exception &e) {
		std::cerr << "[LFSuperRes] OpenCV DNN load exception: " << e.what()
				  << std::endl;
		return false;
	}
}

// --- 辅助函数 ---

std::string LFSuperRes::getModelNameFromMethod(Method method) const {
	switch (method) {
		case Method::ESPCN:
			return "espcn";
		case Method::FSRCNN:
			return "fsrcnn";
		// 如果有 EDSR 等其他模型可以在此添加
		default:
			return "";
	}
}

int LFSuperRes::MethodToInterFlag(Method method) const {
	switch (method) {
		case Method::NEAREST:
			return cv::INTER_NEAREST;
		case Method::CUBIC:
			return cv::INTER_CUBIC;
		case Method::LANCZOS:
			return cv::INTER_LANCZOS4;
		default:
			return cv::INTER_LINEAR;
	}
}

bool LFSuperRes::isDeepLearningMethod(Method method) const {
	return (method == Method::ESPCN || method == Method::FSRCNN
			|| method == Method::DISTGSSR);
}