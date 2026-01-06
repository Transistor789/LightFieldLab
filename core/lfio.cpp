#include "lfio.h"

#include "raw_decode.h"
#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <openssl/sha.h>
#include <string>

LFIO::LFIO() { cv::setNumThreads(cv::getNumberOfCPUs()); }

cv::Mat LFIO::readStandardImage(const std::string &path) {
	// 读取普通图片 (Unchanged 以保留可能的 Alpha 通道或原始深度)
	cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (img.empty()) {
		std::cerr << "[LFIO] Error: Failed to load standard image: " << path
				  << std::endl;
		return cv::Mat();
	}
	// TODO
	// return gamma_convert(img, false);
	return img;
}

cv::Mat LFIO::readLFP(const std::string &path, json *j) {
	RawDecoder decoder;
	cv::Mat img = decoder.decode(path);

	if (!img.empty() && j != nullptr) {
		*j = decoder.lfp;
	}
	return img;
}

std::shared_ptr<LFData> LFIO::readSAI(const std::string &path) {
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
		// 构建全路径
		std::string full_path = path + "/" + filenames[i];

		// A. 读取 (IO + 解码)
		temp[i] = cv::imread(full_path, cv::IMREAD_COLOR);
		// cv::Mat raw_img = cv::imread(full_path, cv::IMREAD_COLOR);

		// 鲁棒性检查
		// 		if (raw_img.empty()) {
		// // 在多线程里打印要注意，最好用原子操作或忽略，这里简单打印
		// #pragma omp critical
		// 			std::cerr << "[Warning] Failed to read: " << filenames[i]
		// 					  << std::endl;
		// 			continue;
		// 		}

		// B. 转换 (计算)
		// 【关键优化】：读完立刻转！
		// 此时 raw_img 的像素数据还在 CPU 缓存里，convertTo 速度极快
		// raw_img.convertTo(temp[i], CV_32FC(raw_img.channels()), 1.0 / 255.0);

		// raw_img 在这里析构，释放 8-bit 内存，降低峰值内存占用
	}

	return std::make_shared<LFData>(std::move(temp));
}

bool LFIO::saveLookUpTables(const std::string &path,
							const std::vector<cv::Mat> &maps, int winSize) {
	if (maps.empty())
		return false;

	// [安全检查] 确保 maps 数量与 winSize 对应
	// 每个 View 有 X 和 Y 两张表，所以总数应该是 winSize^2 * 2
	if (maps.size() != (size_t)winSize * winSize * 2) {
		std::cerr << "Error: Maps count (" << maps.size()
				  << ") does not match winSize (" << winSize << ")"
				  << std::endl;
		return false;
	}

	// [安全检查] 确保是 CV_32F 类型
	if (maps[0].type() != CV_32FC1) {
		std::cerr << "Error: LFIO only supports CV_32FC1 maps." << std::endl;
		return false;
	}

	std::ofstream out(path, std::ios::binary);
	if (!out.is_open())
		return false;

	int rows = maps[0].rows;
	int cols = maps[0].cols;

	// 1. 写入头
	out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
	out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
	out.write(reinterpret_cast<const char *>(&winSize), sizeof(int));

	// 2. 批量写入
	for (const auto &mat : maps) {
		// 二次校验尺寸和类型
		if (mat.rows != rows || mat.cols != cols || mat.type() != CV_32FC1) {
			std::cerr << "Error: Inconsistent map size or type found."
					  << std::endl;
			out.close();
			return false;
		}

		if (mat.isContinuous()) {
			out.write(reinterpret_cast<const char *>(mat.ptr<float>(0)),
					  rows * cols * sizeof(float));
		} else {
			for (int r = 0; r < rows; ++r) {
				out.write(reinterpret_cast<const char *>(mat.ptr<float>(r)),
						  cols * sizeof(float));
			}
		}
	}

	// 检查写入是否成功（磁盘满等情况）
	if (!out) {
		std::cerr << "Error: Write failed (disk full?)" << std::endl;
		return false;
	}

	out.close();
	return true;
}

// 【重要修正】注意这里的 int &outWinSize
bool LFIO::loadLookUpTables(const std::string &path, std::vector<cv::Mat> &maps,
							int &outWinSize) {
	std::ifstream in(path, std::ios::binary);
	if (!in.is_open()) {
		std::cerr << "Error: Cannot open " << path << std::endl;
		return false;
	}

	// 1. 读取头
	int rows, cols, winSize;
	if (!in.read(reinterpret_cast<char *>(&rows), sizeof(int))
		|| !in.read(reinterpret_cast<char *>(&cols), sizeof(int))
		|| !in.read(reinterpret_cast<char *>(&winSize), sizeof(int))) {
		return false;
	}

	// 赋值给引用参数，传出结果
	outWinSize = winSize;

	// 预期读取的矩阵数量
	int totalMats = winSize * winSize * 2;

	maps.clear();
	maps.reserve(totalMats);

	size_t mapByteSize = (size_t)rows * cols * sizeof(float);

	// 2. 循环读取
	for (int i = 0; i < totalMats; ++i) {
		// 创建新矩阵 (OpenCV构造函数创建的矩阵默认是连续的)
		cv::Mat map(rows, cols, CV_32FC1);

		// 直接读取到 data 指针
		in.read(reinterpret_cast<char *>(map.data), mapByteSize);

		// 检查文件是否提前结束
		if (!in) {
			std::cerr << "Error: Unexpected EOF at map index " << i
					  << std::endl;
			return false;
		}

		maps.push_back(map);
	}

	in.close();
	std::cout << "Loaded LUTs: " << rows << "x" << cols
			  << ", WinSize=" << winSize << ", Total Mats=" << maps.size()
			  << std::endl;
	return true;
}