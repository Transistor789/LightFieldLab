#include "trtcli.h"

#include <format>

int main() {
	TrtCliConfig config;

	config.inputName = "input";
	config.optShape = "1x1x780x540";
	config.trtexecPath = "trtexec";
	config.onnxPath = "../data/DistgSSR_2xSR_5x5_780x540.onnx";
	config.enginePath = "../data/DistgSSR_2xSR_5x5_780x540.engine";

	config.fp16 = true; // 开启 FP16

	// --- 执行构建 ---
	if (TrtCli::build(config)) {
		// 成功后，去调用 deploy()
		// deploy(config.enginePath);
		std::cout << "Ready to deploy!" << std::endl;
	} else {
		std::cerr << "Build failed, stopping program." << std::endl;
		return -1;
	}

	return 0;
}