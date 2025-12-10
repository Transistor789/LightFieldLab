#include "config.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
	try {
		fs::create_directories("../data");
		std::string configPath = "../data/config.json";

		if (fs::exists(configPath)) {
			std::cout << "Loading config from: " << configPath << std::endl;
			Config::Get().readParams(configPath);
		} else {
			std::cout << "Config not found. Using defaults.\n";
			Config::Get().saveParams(configPath);
		}

		// 打印
		std::cout << "\n--- Current Config ---\n";
		std::cout
			<< "lfp_path: "
			<< Config::Get().app_cfg()[Config::LFP_PATH].get<std::string>()
			<< "\n"; // string
		std::cout << "lfp_path: " << Config::Get().app_cfg()[Config::LFP_PATH]
				  << "\n"; // json
		std::cout << "lfp_path: "
				  << Config::Get().app_cfg().at(Config::LFP_PATH)
				  << "\n"; // json
		std::cout << "lfp_path: " << Config::Get().app_cfg()[Config::LFP_PATH]
				  << "\n"; // json
		std::cout << "opt_refo: "
				  << (Config::Get().app_cfg().at(Config::OPT_REFO) ? "true"
																   : "false")
				  << "\n";

		// 赋值
		// 支持布尔、整数、浮点、json自身隐式转换，string需要手动get<std::string>()
		std::string lfp_path = Config::Get().app_cfg()[Config::LFP_PATH];
		std::cout << lfp_path << " " << typeid(lfp_path).name() << std::endl;
		bool opt_refo = Config::Get().app_cfg()[Config::OPT_REFO];
		std::cout << opt_refo << " " << typeid(opt_refo).name() << std::endl;

		// 修改为 true（布尔值）
		Config::Get().app_cfg()[Config::OPT_REFO] =
			true; // 注意：传的是 json(true)
		std::cout << "Set opt_refo to true.\n";

		// 保存
		Config::Get().saveParams(configPath);
		std::cout << "Saved to: " << configPath << "\n";

	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}