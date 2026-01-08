#include "califinder.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using json = nlohmann::json;

CaliFinder::CaliFinder(const std::string &cali_dir_root)
	: _cali_root(cali_dir_root) {}

// [新增] 核心查找逻辑：纯粹的路径匹配，没有任何 LFP 文件读取操作
std::string CaliFinder::findPath(const std::string &serial,
								 const std::string &geo_ref) {
	if (serial.empty() || geo_ref.empty()) {
		return "";
	}

	// 1. 确定搜索目录 (Root/Serial/)
	fs::path target_dir = fs::path(_cali_root);
	target_dir /= serial;

	// 2. 寻找 Manifest 文件
	fs::path manifest_path = target_dir / "cal_file_manifest.json";

	// 容错：如果序列号文件夹不存在，尝试在根目录找（兼容扁平目录结构）
	if (!fs::exists(manifest_path)) {
		manifest_path = fs::path(_cali_root) / "cal_file_manifest.json";
		if (!fs::exists(manifest_path)) {
			// std::cerr << "[CaliFinder] Manifest not found." << std::endl;
			return "";
		}
		target_dir = fs::path(_cali_root);
	}

	// 3. 在清单中匹配 Hash
	std::string raw_filename =
		searchInManifest(manifest_path.string(), geo_ref);
	if (raw_filename.empty()) {
		std::cerr << "[CaliFinder] No matching ref found in manifest."
				  << std::endl;
		return "";
	}

	// 4. 构造并验证最终路径
	fs::path original_path = target_dir / raw_filename;
	std::string stem = original_path.stem().string(); // "MOD_0015"

	// 优先列表：.RAW > .raw > 原文件名 (GCT)
	std::vector<std::string> extensions = {".RAW", ".raw"};

	for (const auto &ext : extensions) {
		fs::path try_path = target_dir / (stem + ext);
		if (fs::exists(try_path)) {
			return try_path.string(); // 找到 MOD_0015.RAW，直接返回
		}
	}

	// 如果都没找到 RAW，再检查原本的文件 (GCT) 是否存在作为兜底，或者直接报错
	if (fs::exists(original_path)) {
		// 你可以决定这里是返回 GCT 还是返回空
		// 通常白板必须是图像，返回 GCT 没用，建议记录警告
		std::cerr << "Warning: Found GCT but missing RAW image." << std::endl;
		return original_path.string();
	}

	return "";
}

// [修改] 兼容接口：先读头，再调用 findPath
std::string CaliFinder::findWhiteImage(const std::string &lfp_path) {
	if (!fs::exists(lfp_path))
		return "";

	// 产生 IO 开销
	json lfp_json = readLfpHeader(lfp_path);
	if (lfp_json.empty())
		return "";

	std::string serial;
	std::string geo_ref;

	try {
		// 这里的逻辑就是我们在 RawDecoder::filter_lfp_json 里做的事
		// 提取 serial
		if (lfp_json.contains("camera")
			&& lfp_json["camera"].contains("serialNumber")) {
			serial = lfp_json["camera"]["serialNumber"].get<std::string>();
		}
		// 提取 geo_ref (为了简洁，这里省略了详细的 Gen1/Gen2 判断代码，
		// 实际应复用 RawDecoder 的逻辑或保留你原有的复杂提取逻辑)
		// ... (保留你原有的提取代码) ...
	} catch (...) {
	}

	// 复用核心逻辑
	return findPath(serial, geo_ref);
}

nlohmann::json CaliFinder::readLfpHeader(const std::string &path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		return {};

	// LFP 文件结构通常是:
	// [12 bytes Header] -> [4 bytes Length of Metadata] -> [JSON Metadata] ->
	// [Null] -> [Binary Data] 或者是纯文本 JSON 开始
	// (LFR)。为了简单起见，我们尝试读取头部。

	// 简化的读取策略：读取前 N 字节寻找 JSON 结束符 '}'
	// 注意：严谨的做法是解析 LFP 头部协议，这里使用类似 Python 库的简化逻辑
	// 或者利用 nlohmann::json 的宽容解析

	try {
		// Lytro 文件头处理比较繁琐，这里假设已经有现成的 LFIO::ReadLFP
		// 可以复用， 或者我们尝试直接把文件当文本读，直到第一个 NULL 字节。
		// 下面是一个基于 LFP 规范的读取：

		char magic[12];
		file.read(magic, 12); // Lytro Magic Number

		// 读取元数据长度 (大端序)
		unsigned char len_buf[4];
		file.read((char *)len_buf, 4);
		uint32_t meta_len = (len_buf[0] << 24) | (len_buf[1] << 16)
							| (len_buf[2] << 8) | len_buf[3];

		if (meta_len > 0 && meta_len < 100 * 1024 * 1024) { // 简单的合理性检查
			std::vector<char> buffer(meta_len);
			file.read(buffer.data(), meta_len);
			// 找到第一个 null 截断，因为 json parser 不喜欢尾部垃圾数据
			std::string json_str(buffer.data(), meta_len);
			size_t null_pos = json_str.find('\0');
			if (null_pos != std::string::npos) {
				json_str.resize(null_pos);
			}
			return json::parse(json_str);
		}
	} catch (...) {
		// 如果二进制解析失败，尝试直接作为纯文本解析 (兼容某些解包后的格式)
		file.clear();
		file.seekg(0);
		try {
			json j;
			file >> j;
			return j;
		} catch (...) {
		}
	}

	return {};
}

std::string CaliFinder::searchInManifest(const std::string &manifest_path,
										 const std::string &target_ref) {
	std::ifstream file(manifest_path);
	if (!file.is_open())
		return "";

	try {
		json j;
		file >> j;

		// 遍历清单
		// Illum 结构: "calibrationFiles": [ { "hash": "xxx", "name":
		// "MOD_xxx.GCT" }, ... ]
		if (j.contains("calibrationFiles")
			&& j["calibrationFiles"].is_array()) {
			for (const auto &item : j["calibrationFiles"]) {
				if (item.contains("hash") && item["hash"] == target_ref) {
					if (item.contains("name")) {
						return item["name"].get<std::string>();
					}
				}
			}
		}
		// F01 结构: "frame": { "imageRef": "xxx" } 这种通常是一对一，比较少见
	} catch (...) {
		return "";
	}
	return "";
}