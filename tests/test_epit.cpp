#include "epit.h"
#include "lfio.h"
#include "utils.h"

#include <format>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
	auto lf = LFIO::read_sai("../data/bedroom");

	EPIT ssr;

	// 图片路径
	const std::string SRC_IMG_DIR = "../data/bedroom/";
	const std::string SRC_IMG_PREFIX = "input_Cam";
	int target_scale = 2;
	int target_patch = 64;
	bool center_only = true;

	ssr.setScale(target_scale);
	ssr.setPatchSize(target_patch);
	ssr.setPadding(8);
	ssr.setCenterOnly(center_only);

	// 构造 Engine 路径 (假设你有这个文件)
	std::string enginePath2x =
		std::format("../data/EPIT_2x_1x1x5x5x{}x{}_FP32.engine", target_patch,
					target_patch);
	std::string enginePath4x =
		std::format("../data/EPIT_2x_1x1x5x5x{}x{}_FP32.engine", target_patch,
					target_patch);

	ssr.readEngine(enginePath4x);

	Timer timer;
	auto result = ssr.run(lf->data);
	timer.stop();
	timer.print_elapsed_ms("center_only = " + std::to_string(center_only));
	std::cout << "result.size() = " << result.size() << std::endl;
	cv::imshow("img1", result[result.size() / 2]);

	center_only = !center_only;
	ssr.setCenterOnly(center_only);
	timer.start();
	result = ssr.run(lf->data);
	timer.stop();
	timer.print_elapsed_ms("center_only = " + std::to_string(center_only));
	std::cout << "result.size() = " << result.size() << std::endl;
	cv::imshow("img2", result[result.size() / 2]);
	cv::waitKey();

	return 0;
}