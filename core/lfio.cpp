#include "lfio.h"

#include "califinder.h"
#include "raw_decode.h"
#include "utils.h" // 假设有通用工具

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

cv::Mat LFIO::ReadStandardImage(const std::string &path) {
	cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (img.empty()) {
		std::cerr << "[LFIO] Error: Failed to load standard image: " << path << std::endl;
		return cv::Mat();
	}
	return img;
}

cv::Mat LFIO::ReadLFP(const std::string &path, json &outMetadata) {
	// 使用无状态的 RawDecoder
	RawDecoder decoder;
	cv::Mat img = decoder.DecodeLytro(path, outMetadata);
	return img;
}

cv::Mat LFIO::ReadWhiteImageAuto(const std::string &lfpPath, const std::string &caliDir, json &outWhiteMeta) {
	// 1. 检查标定目录
	if (caliDir.empty())
		return cv::Mat();

	// 2. 预读 LFP 获取 Key (Serial, GeoRef)
	RawDecoder decoder;
	json lfpMeta;
	try {
		decoder.DecodeLytro(lfpPath, lfpMeta);
	} catch (...) {
		return cv::Mat();
	}

	std::string serial, geoRef;
	if (lfpMeta.contains("serial"))
		serial = lfpMeta["serial"];
	if (lfpMeta.contains("geo_ref"))
		geoRef = lfpMeta["geo_ref"];

	if (serial.empty() || geoRef.empty())
		return cv::Mat();

	// 3. 查找路径
	CaliFinder finder(caliDir);
	std::string whitePath = finder.findPath(serial, geoRef);

	if (whitePath.empty())
		return cv::Mat();

	std::cout << "[LFIO] Auto-found white image: " << whitePath << std::endl;

	// 4. 解码
	return decoder.DecodeWhiteImage(whitePath, outWhiteMeta);
}

// =============================================================
// [方式二：手动]
// =============================================================
cv::Mat LFIO::ReadWhiteImageManual(const std::string &whitePath, json &outWhiteMeta) {
	// 简单的参数检查
	if (whitePath.empty())
		return cv::Mat();
	if (!std::filesystem::exists(whitePath)) {
		std::cerr << "[LFIO] Manual white image path not found: " << whitePath << std::endl;
		return cv::Mat();
	}

	// 直接调用解码器，解码器内部会自动处理 .TXT/.json 的寻找和应用
	RawDecoder decoder;
	return decoder.DecodeWhiteImage(whitePath, outWhiteMeta);
}

std::shared_ptr<LFData> LFIO::ReadSAI(const std::string &path) {
	// 1. 检查路径
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("readSAI: file not exist! Path: " + path);
	}

	// 2. 获取文件列表 (这部分必须串行，IO瓶颈且 filesystem
	// 迭代器不支持随机访问)
	std::vector<std::string> filenames;
	filenames.reserve(200); // 预留空间，微小优化
	for (const auto &entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(entry.path().filename().string());
		}
	}

	// 排序确保视角顺序正确
	std::sort(filenames.begin(), filenames.end());

	size_t total_files = filenames.size();
	if (total_files == 0)
		return nullptr;

	// 3. 预分配结果容器 (关键：避免并行时的 push_back 锁竞争)
	std::vector<cv::Mat> temp(total_files);

// ============================================================
// OpenMP 优化核心
// ============================================================
// schedule(dynamic):
// 因为文件大小可能不同，读取解码时间不一，动态调度防止线程空等
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < total_files; ++i) {
		temp[i] = cv::imread(path + "/" + filenames[i], cv::IMREAD_COLOR);
	}

	return std::make_shared<LFData>(std::move(temp));
}

void LFIO::SaveSAI(const std::string &path, std::shared_ptr<LFData> lf) {
	if (!lf || lf->data.empty()) {
		std::cerr << "[LFIO] Error: LFData is null or empty." << std::endl;
		return;
	}
	// 调用重载函数处理内部的 std::vector<cv::Mat>
	SaveSAI(path, lf->data);
}

void LFIO::SaveSAI(const std::string &path, const std::vector<cv::Mat> &lf) {
	namespace fs = std::filesystem;

	// 1. 确保目录存在
	if (!fs::exists(path)) {
		if (!fs::create_directories(path)) {
			std::cerr << "[LFIO] Error: Failed to create directory " << path << std::endl;
			return;
		}
	}

	// 2. 遍历并保存图像
	for (size_t i = 0; i < lf.size(); ++i) {
		// 使用 C++20 std::format 匹配命名规则 input_Cam%03d
		std::string fileName = std::format("input_Cam{:03d}.bmp", i + 1);
		fs::path fullPath = fs::path(path) / fileName;

		try {
			// 保存图像，注意：如果你的图像是 CV_32F 需先转为 CV_8U
			cv::Mat saveImg = lf[i];
			if (saveImg.depth() == CV_32F || saveImg.depth() == CV_64F) {
				// 自动映射到 0-255 范围以保存为 bmp
				saveImg.convertTo(saveImg, CV_8U, 255.0);
			}

			if (!cv::imwrite(fullPath.string(), saveImg)) {
				std::cerr << "[LFIO] Failed to write: " << fullPath << std::endl;
			}
		} catch (const cv::Exception &e) {
			std::cerr << "[LFIO] OpenCV Exception: " << e.what() << std::endl;
		}
	}

	std::cout << std::format("[LFIO] Successfully saved {} SAIs to {}", lf.size(), path) << std::endl;
}

bool LFIO::SaveLookUpTables(const std::string &path, const std::vector<cv::Mat> &maps, int winSize) {
	if (maps.empty())
		return false;

	std::ofstream out(path, std::ios::binary);
	if (!out.is_open())
		return false;

	int rows = maps[0].rows;
	int cols = maps[0].cols;
	int count = static_cast<int>(maps.size());

	// 1. 严格写入 4 个 int 的头信息
	out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
	out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
	out.write(reinterpret_cast<const char *>(&winSize), sizeof(int));
	out.write(reinterpret_cast<const char *>(&count), sizeof(int));

	// 2. 批量写入数据
	for (const auto &mat : maps) {
		// 直接写入连续内存
		out.write(reinterpret_cast<const char *>(mat.data), mat.total() * mat.elemSize());
	}

	out.flush();
	out.close();
	return true;
}

bool LFIO::LoadLookUpTables(const std::string &path, std::vector<cv::Mat> &maps, int &outWinSize) {
	// 1. 初始化所有变量为 0，防止出现 0xCCCC... 的天文数字
	int rows = 0, cols = 0, winSize = 0, count = 0;

	std::ifstream in(path, std::ios::binary);
	if (!in.is_open()) {
		std::cerr << "[LFIO] Error: File not found at " << std::filesystem::absolute(path) << std::endl;
		return false;
	}

	// 2. 顺序读取 4 个头信息并校验
	if (!in.read((char *)&rows, 4) || !in.read((char *)&cols, 4) || !in.read((char *)&winSize, 4)
		|| !in.read((char *)&count, 4)) {
		std::cerr << "[LFIO] Error: Header incomplete." << std::endl;
		return false;
	}

	// 3. 严格校验：如果 rows 是 430 而不是你预期的 542，说明文件版本不对
	if (rows <= 0 || count <= 0 || count > 2000) {
		std::cerr << "[LFIO] Format Error. File Rows: " << rows << " Count: " << count << std::endl;
		return false;
	}

	outWinSize = winSize;
	maps.clear();
	maps.reserve(count);

	size_t mapByteSize = (size_t)rows * cols * sizeof(float);
	for (int i = 0; i < count; ++i) {
		cv::Mat map(rows, cols, CV_32FC1);
		if (!in.read((char *)map.data, mapByteSize))
			return false;
		maps.push_back(map);
	}
	in.close();
	return true;
}