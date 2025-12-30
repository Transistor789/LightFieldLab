#include "distgssr.h"
#include "lfio.h"
#include "utils.h"

#include <format>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void test() {
	std::cout
		<< "\n================ TensorRT 10 Environment Check ================"
		<< std::endl;

	// --- 1. 检查版本 ---
	std::cout << "TensorRT Version : " << NV_TENSORRT_MAJOR << "."
			  << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;

	// --- 2. 创建 Logger ---
	TRTLogger logger;

	// --- 3. 创建 Builder (使用智能指针 RAII 管理，防止内存泄露和析构顺序错误)
	// --- 自定义删除器不是必须的，因为 IBuilder 有虚析构函数，但 unique_ptr
	// 需要指定类型
	auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
	if (!builder) {
		std::cerr << "[Error] Failed to create Builder! Check your "
					 "CUDA/TensorRT installation."
				  << std::endl;
		return;
	}
	std::cout << "Builder          : Created successfully." << std::endl;

	// --- 4. 创建 Network (显式 Batch) ---
	const auto explicitBatch =
		1U << static_cast<uint32_t>(
			NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = std::unique_ptr<INetworkDefinition>(
		builder->createNetworkV2(explicitBatch));
	if (!network) {
		std::cerr << "[Error] Failed to create Network!" << std::endl;
		return;
	}
	std::cout << "Network          : Created successfully." << std::endl;

	// --- 5. 创建 Config ---
	auto config =
		std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
	if (!config) {
		std::cerr << "[Error] Failed to create Config!" << std::endl;
		return;
	}
	std::cout << "Config           : Created successfully." << std::endl;

	// --- 6. 检查 GPU 硬件支持 (替代被弃用的 platformHasFastFp16) ---
	int deviceId = 0;
	cudaGetDevice(&deviceId);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceId);

	std::cout << "GPU Device       : " << props.name << std::endl;
	std::cout << "Compute Cap      : " << props.major << "." << props.minor
			  << std::endl;

	// TRT 10 默认会自动使用最佳精度。这里我们手动检查一下硬件是否给力。
	// 算力 >= 7.0 (Volta/Turing/Ampere/Hopper) 通常都有 Tensor Core
	if (props.major >= 7) {
		std::cout << "FP16 Acceleration: [SUPPORTED] (Tensor Cores available)"
				  << std::endl;
	} else if (props.major == 6 && props.minor == 0) {
		std::cout << "FP16 Acceleration: [SUPPORTED] (Pascal P100 mode)"
				  << std::endl;
	} else {
		std::cout << "FP16 Acceleration: [LIMITED] (Old GPU, might be slow)"
				  << std::endl;
	}

	std::cout << "================ Test Passed Successfully! ================\n"
			  << std::endl;
}

int main() {
	// test();
	auto lf = LFIO::readSAI("../data/bedroom");

	DistgSSR ssr;

	// 图片路径
	const std::string SRC_IMG_DIR = "../data/bedroom/";
	const std::string SRC_IMG_PREFIX = "input_Cam";
	int target_scale = 4;
	int target_patch = 128;
	bool center_only = true;

	ssr.setScale(target_scale);
	ssr.setPatchSize(target_patch);
	ssr.setPadding(8);
	ssr.setCenterOnly(center_only);

	ssr.readEngine(std::format("../data/DistgSSR_{}x_1x1x{}x{}_FP16.engine",
							   target_scale, 5 * target_patch,
							   5 * target_patch));

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