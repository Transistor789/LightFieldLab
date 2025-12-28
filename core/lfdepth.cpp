#include "lfdepth.h"

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
	// 1. 检查是否需要重新加载
	// 条件：模型未加载 OR 参数发生了变更
	bool needReload = !distg_.isEngineLoaded()
					  || (m_loadedAngRes != m_targetAngRes)
					  || (m_loadedPatchSize != m_targetPatchSize);

	if (!needReload) {
		return true; // 状态一致，无需操作
	}

	// 2. 生成模型路径
	// 假设模型都在 data/ 目录下，命名规则如: DistgDisp_9x9_128_FP16.engine
	std::string modelPath = getModelPath(m_targetAngRes, m_targetPatchSize);

	// 3. 加载模型
	// 注意：这里需要捕获可能的异常，或者检查文件是否存在
	try {
		distg_.readEngine(modelPath);

		// 4. 同步参数给底层算法
		distg_.setAngRes(m_targetAngRes);
		distg_.setPatchSize(m_targetPatchSize);
		// distg_.setPadding(...); // 如果 padding 跟 patchSize
		// 有关，也需要在这里更新

		// 5. 更新“已加载”状态
		m_loadedAngRes = m_targetAngRes;
		m_loadedPatchSize = m_targetPatchSize;

		std::cout << "[LFDisp] Model loaded successfully: " << modelPath
				  << std::endl;
		return true;

	} catch (const std::exception &e) {
		std::cerr << "[LFDisp] Failed to load model: " << modelPath
				  << "\nReason: " << e.what() << std::endl;
		return false;
	}
}

std::string LFDisp::getModelPath(int angRes, int patchSize) const {
	// C++20 写法
	return std::format("data/DistgDisp_{}x{}_{}_FP16.engine", angRes, angRes,
					   patchSize);
}