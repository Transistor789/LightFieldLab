#include "lfdepth.h"

#include "lfparams.h"

bool LFDisp::depth(const std::vector<cv::Mat> &views) {
	if (views.empty())
		return false;

	// --- 【新增】 运行前检查 ---
	if (!checkAndLoadModel()) {
		std::cerr << "[LFDisp] Abort: Model check failed." << std::endl;
		return false;
	}

	// 调用核心算法 (此时模型一定已就绪)
	// 注意：distg_.run 返回的是 vector，根据你之前的代码，这里取第一个
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
	// 复用 getGrayVisual 的归一化逻辑
	cv::Mat gray = getGrayVisual();
	if (gray.empty())
		return cv::Mat();

	cv::Mat colorMap;
	// 应用 Jet Colormap (经典的彩虹色：蓝-青-黄-红)
	cv::applyColorMap(gray, colorMap, cv::COLORMAP_JET);
	return colorMap;
}

cv::Mat LFDisp::getPlasmaVisual() const {
	cv::Mat gray = getGrayVisual();
	if (gray.empty())
		return cv::Mat();

	cv::Mat colorMap;
	// Plasma 是一种感知更均匀的色阶 (紫-橙-黄)，看起来更“现代”
	cv::applyColorMap(gray, colorMap, cv::COLORMAP_PLASMA);
	return colorMap;
}

bool LFDisp::checkAndLoadModel() {
	// 【修复步骤 2】将“类型是否变化”加入判断条件
	bool typeChanged = (m_loadedType != type);

	// 重新加载的条件（满足任意一条即可）：
	// 1. 引擎本身未加载
	// 2. 角度分辨率变了
	// 3. PatchSize 变了
	// 4. 【新增】算法类型变了 (DistgSSR <-> OACC)
	bool needReload = !distg_.isEngineLoaded()
					  || (m_loadedAngRes != m_targetAngRes)
					  || (m_loadedPatchSize != m_targetPatchSize)
					  || typeChanged; // <--- 关键修改

	if (!needReload) {
		return true; // 状态完全一致，无需重新加载
	}

	// 生成模型路径 (getModelPath 内部已经用了最新的 type，所以路径是对的)
	std::string modelPath = getModelPath(m_targetAngRes, m_targetPatchSize);

	// 打印调试信息，方便你看清楚是不是真的切了
	std::cout << "[LFDisp] Model change detected. Reloading..." << std::endl;
	std::cout << "   Target Type: "
			  << (type == LFParamsDE::Type::DistgSSR ? "DistgSSR" : "OACC")
			  << std::endl;
	std::cout << "   Target Path: " << modelPath << std::endl;

	try {
		distg_.readEngine(modelPath);

		// 同步参数给底层算法
		distg_.setAngRes(m_targetAngRes);
		distg_.setPatchSize(m_targetPatchSize);

		// 【修复步骤 3】更新所有“已加载”状态，包括 Type
		m_loadedAngRes = m_targetAngRes;
		m_loadedPatchSize = m_targetPatchSize;
		m_loadedType = type; // <--- 关键：记录下当前加载的是 OACC 还是 DistgSSR

		std::cout << "[LFDisp] Model loaded successfully." << std::endl;
		return true;

	} catch (const std::exception &e) {
		std::cerr << "[LFDisp] Failed to load model: " << modelPath
				  << "\nReason: " << e.what() << std::endl;

		// 如果加载失败，重置状态，强制下次重试
		m_loadedAngRes = -1;
		return false;
	}
}

std::string LFDisp::getModelPath(int angRes, int patchSize) const {
	std::string path;

#ifdef _WIN32
	if (type == LFParamsDE::Type::DistgSSR) {
		path = std::format("data/DistgDisp_{}x{}_{}_FP16_Windows.engine",
						   angRes, angRes, patchSize);
	} else if (type == LFParamsDE::Type::OACC) {
		path = std::format("data/OACC-Net_{}x{}_{}_FP16_Windows.engine", angRes,
						   angRes, patchSize);
	}

#elif __linux__
	if (type == LFParamsDE::Type::DistgSSR) {
		path = std::format("data/DistgDisp_{}x{}_{}_FP16_Linux.engine", angRes,
						   angRes, patchSize);
	} else if (type == LFParamsDE::Type::OACC) {
		path = std::format("data/OACC-Net_{}x{}_{}_FP16_Linux.engine", angRes,
						   angRes, patchSize);
	}
#endif
	return path;
}