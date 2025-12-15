#ifndef BIN_IO_H
#define BIN_IO_H

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace BinIO {

// ==========================================
// 1. 基础数据类型 (int, float, struct)
// ==========================================

// 保存基础类型
template <typename T>
void save(const std::string &path, const T &data) {
	std::ofstream out(path, std::ios::binary);
	if (!out.is_open()) {
		std::cerr << "[BinIO] Error: Cannot open " << path << " for write."
				  << std::endl;
		return;
	}
	out.write(reinterpret_cast<const char *>(&data), sizeof(T));
	out.close();
}

// 读取基础类型
template <typename T>
void load(const std::string &path, T &data) {
	std::ifstream in(path, std::ios::binary);
	if (!in.is_open()) {
		std::cerr << "[BinIO] Error: Cannot open " << path << " for read."
				  << std::endl;
		return;
	}
	in.read(reinterpret_cast<char *>(&data), sizeof(T));
	in.close();
}

// ==========================================
// 2. std::vector<T> 的特化支持
// ==========================================

// 保存 Vector
template <typename T>
void save(const std::string &path, const std::vector<T> &vec) {
	std::ofstream out(path, std::ios::binary);
	if (!out.is_open())
		return;

	// 1. 先存元素数量 (使用 uint64_t 确保 64位兼容)
	uint64_t size = vec.size();
	out.write(reinterpret_cast<const char *>(&size), sizeof(size));

	// 2. 存数据块
	if (size > 0) {
		out.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(T));
	}
	out.close();
}

// 读取 Vector
template <typename T>
void load(const std::string &path, std::vector<T> &vec) {
	std::ifstream in(path, std::ios::binary);
	if (!in.is_open())
		return;

	// 1. 先读元素数量
	uint64_t size = 0;
	in.read(reinterpret_cast<char *>(&size), sizeof(size));

	// 2. 调整 vector 大小并读数据
	vec.resize(size);
	if (size > 0) {
		in.read(reinterpret_cast<char *>(vec.data()), size * sizeof(T));
	}
	in.close();
}

// ==========================================
// 3. cv::Mat 的特化支持
// ==========================================

// 保存 cv::Mat
inline void save(const std::string &path, const cv::Mat &mat) {
	std::ofstream out(path, std::ios::binary);
	if (!out.is_open())
		return;

	if (mat.empty()) {
		int rows = 0;
		out.write((char *)&rows, sizeof(int));
		return;
	}

	// 1. 存元数据: Rows, Cols, Type
	int head[3] = {mat.rows, mat.cols, mat.type()};
	out.write(reinterpret_cast<const char *>(head), sizeof(head));

	// 2. 存像素数据
	// 注意：如果是 ROI 裁剪出来的 Mat，内存可能不连续，需要逐行写入
	if (mat.isContinuous()) {
		out.write(reinterpret_cast<const char *>(mat.data),
				  mat.total() * mat.elemSize());
	} else {
		// 如果不连续，逐行写入
		size_t rowSize = mat.cols * mat.elemSize();
		for (int i = 0; i < mat.rows; ++i) {
			out.write(reinterpret_cast<const char *>(mat.ptr(i)), rowSize);
		}
	}
	out.close();
}

// 读取 cv::Mat
inline void load(const std::string &path, cv::Mat &mat) {
	std::ifstream in(path, std::ios::binary);
	if (!in.is_open())
		return;

	// 1. 读元数据
	int head[3]; // rows, cols, type
	in.read(reinterpret_cast<char *>(head), sizeof(head));

	if (head[0] == 0) {
		mat = cv::Mat();
		return;
	}

	// 2. 重建矩阵 (create 会自动处理内存分配)
	mat.create(head[0], head[1], head[2]);

	// 3. 读像素数据 (新创建的 Mat 肯定是连续的)
	in.read(reinterpret_cast<char *>(mat.data), mat.total() * mat.elemSize());
	in.close();
}
} // namespace BinIO

#endif // BIN_IO_H