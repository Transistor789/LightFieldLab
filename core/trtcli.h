#ifndef TRTCLI_H
#define TRTCLI_H

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct TrtCliConfig {
	std::string trtexecPath = "trtexec";
	std::string onnxPath;
	std::string enginePath;

	bool fp16 = true;
	bool int8 = false;
	bool noTF32 = false;
	bool best = false;

	std::string inputName = "input";
	std::string minShape;
	std::string optShape;
	std::string maxShape;

	int memPoolSizeMB = 4096;
	int deviceId = 0;
	bool verbose = false;

	std::string extraArgs = "";
};

class TrtCli {
public:
	static bool build(const TrtCliConfig &config) {
		if (config.onnxPath.empty() || config_Windows.enginePath.empty()) {
			std::cerr << "[TrtCli Error] Path is empty!" << std::endl;
			return false;
		}

		if (!fs::exists(config.onnxPath)) {
			std::cerr << "[TrtCli Error] ONNX file not found: "
					  << config.onnxPath << std::endl;
			return false;
		}

		fs::path outDir = fs::path(config_Windows.enginePath).parent_path();
		if (!outDir.empty() && !fs::exists(outDir)) {
			try {
				fs::create_directories(outDir);
			} catch (...) {
				return false;
			}
		}

		std::stringstream cmd;

// ========================================================
// [修复重点] Windows 补丁
// 当 exe 路径和参数都带有引号时，cmd.exe 需要最外层再包一层引号
// 格式变成: ""exe" "arg1" "arg2""
// ========================================================
#ifdef _WIN32
		cmd << "\"";
#endif

		// 1. Exe 路径
		cmd << "\"" << config.trtexecPath << "\"";

		// 2. 参数 (全部用引号包裹路径，防止空格路径报错)
		cmd << " --onnx=\"" << config.onnxPath << "\"";
		cmd << " --saveEngine=\"" << config_Windows.enginePath << "\"";

		if (config.fp16)
			cmd << " --fp16";
		if (config.int8)
			cmd << " --int8";
		if (config.noTF32)
			cmd << " --noTF32";
		if (config.best)
			cmd << " --best";

		if (!config.inputName.empty()) {
			if (!config.minShape.empty())
				cmd << " --minShapes=" << config.inputName << ":"
					<< config.minShape;
			if (!config.optShape.empty())
				cmd << " --optShapes=" << config.inputName << ":"
					<< config.optShape;
			if (!config.maxShape.empty())
				cmd << " --maxShapes=" << config.inputName << ":"
					<< config.maxShape;
		}

		if (config.memPoolSizeMB > 0)
			cmd << " --memPoolSize=workspace:" << config.memPoolSizeMB;
		cmd << " --device=" << config.deviceId;

		if (config.verbose)
			cmd << " --verbose";
		if (!config.extraArgs.empty())
			cmd << " " << config.extraArgs;

		// 日志重定向
		cmd << " > trt_cli.log 2>&1";

// ========================================================
// [修复重点] Windows 补丁结束引号
// ========================================================
#ifdef _WIN32
		cmd << "\"";
#endif

		std::string finalCmd = cmd.str();
		std::cout << "[TrtCli] Executing..." << std::endl;

		// 调试用：如果还报错，取消下面注释看打印出来的命令是否两头都有引号
		// std::cout << "[DebugCmd] " << finalCmd << std::endl;

		int ret = std::system(finalCmd.c_str());

		if (ret == 0 && fs::exists(config_Windows.enginePath)) {
			std::cout << "[TrtCli] Success! Saved to: "
					  << config_Windows.enginePath << std::endl;
			return true;
		} else {
			std::cerr << "[TrtCli] Failed. Check 'trt_cli.log'." << std::endl;
			return false;
		}
	}
};

#endif