#include "lfdepth.h"

#include <format> // C++20 std::format
#include <iostream>

// [移除] #include "lfparams.h"

bool LFDisp::depth(const std::vector<cv::Mat> &views, Method method) {
	if (views.empty()) {
		std::cerr << "[LFDisp] Error: Empty views input." << std::endl;
		return false;
	}

	if (views[0].depth() != CV_8U) {
		std::cerr << "[LFDisp] Error: Input views must be 8-bit (CV_8U)."
				  << std::endl;
		return false;
	}

	// --- 智能加载：检查并按需加载模型 ---
	// 将 method 传入，与内部状态比对
	if (!checkAndLoadModel(method)) {
		std::cerr << "[LFDisp] Abort: Model check/load failed." << std::endl;
		return false;
	}

	// 调用核心算法 (此时模型一定已就绪且匹配)
	// 注意：distg_.run 返回 vector，这里取第一个结果
	cv::Mat results = distg_.run(views);

	if (results.empty()) {
		m_rawMap = cv::Mat();
		return false;
	}

	m_rawMap = results;
	return true;
}

const cv::Mat &LFDisp::getRawMap() const { return m_rawMap; }

cv::Mat LFDisp::getGrayVisual() const {
	if (m_rawMap.empty())
		return cv::Mat();

	cv::Mat normMap;
	// 归一化: 将 float 的 min-max 映射到 0-255
	cv::normalize(m_rawMap, normMap, 0, 255, cv::NORM_MINMAX, CV_8U);
	return normMap;
}

cv::Mat LFDisp::getJetVisual() const {
	cv::Mat gray = getGrayVisual();
	if (gray.empty())
		return cv::Mat();

	cv::Mat colorMap;
	// Jet Colormap (蓝-青-黄-红)
	cv::applyColorMap(gray, colorMap, cv::COLORMAP_JET);
	return colorMap;
}

cv::Mat LFDisp::getPlasmaVisual() const {
	cv::Mat gray = getGrayVisual();
	if (gray.empty())
		return cv::Mat();

	cv::Mat colorMap;
	// Plasma (紫-橙-黄)
	cv::applyColorMap(gray, colorMap, cv::COLORMAP_PLASMA);
	return colorMap;
}

bool LFDisp::checkAndLoadModel(Method targetMethod) {
	// 1. 检查状态是否发生变化
	bool methodChanged = (m_loadedMethod != targetMethod);
	bool paramChanged = (m_loadedAngRes != m_targetAngRes)
						|| (m_loadedPatchSize != m_targetPatchSize);
	bool engineNotLoaded = !distg_.isEngineLoaded();

	// 如果所有状态一致且引擎已加载，直接跳过，复用现有模型
	if (!engineNotLoaded && !methodChanged && !paramChanged) {
		return true;
	}

	// 2. 准备加载
	std::string modelPath =
		getModelPath(targetMethod, m_targetAngRes, m_targetPatchSize);

	// 打印调试信息
	std::string methodName =
		(targetMethod == Method::DistgDisp) ? "DistgDisp" : "OACC";
	std::cout << "[LFDisp] Model change detected. Reloading..." << std::endl;
	std::cout << "   Target Method: " << methodName << std::endl;
	std::cout << "   Target Path:   " << modelPath << std::endl;

	try {
		// 读取引擎文件
		distg_.readEngine(modelPath);

		// 同步参数给底层算法
		distg_.setAngRes(m_targetAngRes);
		distg_.setPatchSize(m_targetPatchSize);

		// 3. 更新缓存状态 (关键步骤)
		m_loadedAngRes = m_targetAngRes;
		m_loadedPatchSize = m_targetPatchSize;
		m_loadedMethod = targetMethod; // 记录当前加载的是哪种方法

		std::cout << "[LFDisp] Model loaded successfully." << std::endl;
		return true;

	} catch (const std::exception &e) {
		std::cerr << "[LFDisp] Failed to load model: " << modelPath
				  << "\nReason: " << e.what() << std::endl;

		// 如果加载失败，重置状态为无效，强制下次必须重试
		m_loadedAngRes = -1;
		return false;
	}
}

std::string LFDisp::getModelPath(Method method, int angRes,
								 int patchSize) const {
	// 基础名称前缀
	std::string prefix;
	if (method == Method::DistgDisp) {
		prefix = "DistgDisp";
	} else if (method == Method::OACC) {
		prefix = "OACC-Net";
	} else {
		return ""; // Should not happen
	}

	// 平台后缀
#ifdef _WIN32
	std::string osSuffix = "Windows";
#elif __linux__
	std::string osSuffix = "Linux";
#else
	std::string osSuffix = "Unknown";
#endif

	// 使用 format 拼接路径
	// 格式: data/{Method}_{Ang}x{Ang}_{Patch}_FP16_{OS}.engine
	return std::format("data/{}_{}x{}_{}_FP16_{}.engine", prefix, angRes,
					   angRes, patchSize, osSuffix);
}