// lfcapture.cpp

#include "lfcapture.h"

#include "USBConfiguration.h"

#include <iostream> // 必须包含，否则 std::cout 会报错
#include <opencv2/core/types.hpp>

LFCapture::LFCapture() {
	// 初始化成员变量，防止后面逻辑跳过后变量未初始化
	start_x = 0;
	start_y = 0;
	width = 0;
	height = 0;

	// ================= 修改点 1: 屏蔽 SetUSBConfiguration =================
#if defined(_WIN32) || defined(_WIN64)
	// --- Windows 逻辑 ---
	int ret = SetUSBConfiguration(1920, 1080, 3, 0, 0, 0, false, false);
	if (ret == 0) {
		std::cout << "USB configuration succeeded." << std::endl;
	} else {
		std::cout << "USB configuration failed! Error code: " << ret
				  << std::endl;
	}
#else
	// --- Linux 逻辑 ---
	// Linux 下没有这个库的实现，什么都不做，或者打印一条日志
	std::cout << "[Linux] Skipping SetUSBConfiguration (Not supported)."
			  << std::endl;
#endif

	open(0);
}

cv::Mat LFCapture::getFrame() {
	cv::Mat frame, gray_frame;

	if (!cap.isOpened()) {
		// 如果相机没打开，返回空矩阵防止崩溃
		return cv::Mat();
	}

	cap >> frame;
	if (frame.empty()) {
		return cv::Mat();
	}

	cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

	// 确保裁剪区域在图像范围内，防止崩溃
	cv::Rect roi(start_x, start_y, width, height);
	cv::Rect img_rect(0, 0, gray_frame.cols, gray_frame.rows);

	// 取交集，确保安全
	return gray_frame(roi & img_rect);
}

std::vector<int> LFCapture::getAvailableCameras(int maxSearch) {
	std::vector<int> availableIndices;

	// 尝试打开 0 到 maxSearch 的设备
	// 注意：在 Linux 下这通常比较快，但在 Windows 下可能会有显著延迟
	for (int i = 0; i < maxSearch; ++i) {
		cv::VideoCapture tempCap;
		// 使用 V4L2 后端 (Linux) 或 ANY (自动)
		// 在 Linux 上显式指定 CAP_V4L2 通常更稳健
		tempCap.open(i, cv::CAP_ANY);

		if (tempCap.isOpened()) {
			availableIndices.push_back(i);
			tempCap.release(); // 记得立即释放
		}
	}
	return availableIndices;
}

bool LFCapture::open(int index) {
	if (cap.isOpened()) {
		cap.release();
	}

#if defined(_WIN32) || defined(_WIN64)
	// Windows: 使用 DirectShow 接口
	cap.open(index, cv::CAP_DSHOW);
#else
	// Linux: 使用默认接口 (通常是 V4L2)
	// 如果不需要打开相机，甚至可以在这里注释掉 cap.open
	// cap.open(0);
#endif

	if (cap.isOpened()) {
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

		int cam_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

		// 你的原始逻辑
		width = 1024;
		height = 768;
		start_x = (cam_width - width) / 2;
		start_y = 0;

		// 防止除零或负数 (防御性编程)
		if (start_x < 0)
			start_x = 0;

		std::cout << "[LFCapture] Device " << index << " opened successfully."
				  << std::endl;
		return true;
	} else {
		std::cerr << "[LFCapture] Failed to open device " << index << std::endl;
		return false;
	}
}