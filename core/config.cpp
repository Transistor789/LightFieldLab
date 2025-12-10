// src/config.cpp
#include "config.h"

#include "utils.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

Config::Config() {
	defaultConfigFile_ = "../data/config.json";

	readParams(defaultConfigFile_);
	defaultValues();

	calibs_ = json::object();
	img_meta_ = json::object();
}

void Config::defaultValues() {
	app_cfg_ =
		json{{LFP_PATH, ""},		 {CAL_PATH, ""},	  {CAL_META, ""},
			 {CAL_METH, "grid-fit"}, {SMP_METH, "local"}, {PTC_LENG, 7},
			 {RAN_REFO, {0, 1}},	 {OPT_CALI, false},	  {OPT_VIGN, true},
			 {OPT_LIER, 0},			 {OPT_CONT, false},	  {OPT_COLO, 1},
			 {OPT_AWB_, true},		 {OPT_SAT_, false},	  {OPT_VIEW, 1},
			 {OPT_REFO, true}, // 注意：默认用 bool
			 {OPT_REFI, false},		 {OPT_PFLU, false},	  {OPT_ARTI, true},
			 {OPT_ROTA, 0},			 {OPT_DBUG, 0},		  {OPT_PRNT, 1},
			 {OPT_DPTH, false},		 {DIR_REMO, false}};
	// calibs_ = json::object();
}

void Config::resetValues() { defaultValues(); }

// --- 文件操作 ---
void Config::readParams(const std::string &filePath) {
	std::string path = filePath.empty() ? defaultConfigFile_ : filePath;

	app_cfg_ = readJson(path);
}

void Config::saveParams(const std::string &filePath) {
	std::string path = filePath.empty() ? defaultConfigFile_ : filePath;
	writeJson(path, app_cfg_);
}

void Config::ensureDirExists(const std::string &dirPath) {
	if (!dirPath.empty()) {
		fs::create_directories(dirPath);
	}
}
