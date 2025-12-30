#include "distgdisp.h"
#include "lfio.h"
#include "utils.h"

#include <format> // C++20
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
void test() {
	// 1. 设置参数
	int patchSize = 196; // Engine 的输出尺寸
	int padding = 8;	 // 重叠区域

	// 2. 初始化
	DistgDisp dispenser;
	std::string enginePath =
		std::format("../data/DistgDisp_9x9_{}_FP16.engine", patchSize);
	dispenser.readEngine(enginePath);

	// 【关键】配置类
	dispenser.setPatchSize(patchSize); // 设为 128，内部会自动处理 1152 的输入
	dispenser.setPadding(padding);

	if (!dispenser.isEngineLoaded()) {
		std::cerr << "Failed to load engine." << std::endl;
		return;
	}

	// 3. 读取图像
	auto lf = LFIO::readSAI("../data/bedroom");

	// 4. 执行推理
	std::cout << "Estimating disparity..." << std::endl;
	Timer timer;
	cv::Mat dispMap = dispenser.run(lf->data);
	timer.stop();
	timer.print_elapsed_ms();

	// 5. 显示结果
	if (!dispMap.empty()) {
		cv::Mat dispVis;
		cv::normalize(dispMap, dispVis, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::applyColorMap(dispVis, dispVis, cv::COLORMAP_JET);
		cv::imshow("final_result.png", dispVis);
		cv::waitKey();
	}
}
void patch_test() {
	auto lf = LFIO::readSAI("../data/bedroom");

	DistgDisp disp;
	disp.readEngine("../data/DistgDisp_9x9_196_FP16.engine");
	disp.setPatchSize(196);

	Timer timer;
	disp.run(lf->data);
	timer.stop();
	timer.print_elapsed_ms();

	disp.readEngine("../data/DistgDisp_9x9_128_FP16.engine");
	disp.setPatchSize(128);

	timer.start();
	disp.run(lf->data);
	timer.stop();
	timer.print_elapsed_ms();
}
int main() {
	// test();
	patch_test();

	return 0;
}