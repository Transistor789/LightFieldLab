#ifndef LFDATA_H
#define LFDATA_H

#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

class LFData {
public:
	// 保持 Public 以兼容现有项目依赖
	// (在未来重构中，建议改为 private 并提供 getRows() 等接口)
	int size = 0;
	int rows = 0;
	int cols = 0;
	int height = 0;
	int width = 0;
	int channels = 0;
	int type = 0;

	std::vector<cv::Mat> data;

public:
	// 1. 默认构造
	explicit LFData() = default;

	// 2. 从 vector 拷贝构造 (深拷贝)
	explicit LFData(const std::vector<cv::Mat> &src) {
		validate(src); // 必须校验
		data.reserve(src.size());
		for (const auto &mat : src) {
			data.push_back(mat.clone()); // 确保数据独立
		}
		setParam();
	}

	// 3. 从 vector 移动构造 (极致性能)
	// 用法: LFData lf(std::move(my_temp_vec));
	explicit LFData(std::vector<cv::Mat> &&src) {
		validate(src);
		data = std::move(src); // 零拷贝接管
		setParam();
	}

	// 4. 类拷贝构造 (深拷贝)
	LFData(const LFData &src) {
		if (src.data.empty())
			return;
		data.reserve(src.data.size());
		for (const auto &mat : src.data) {
			data.emplace_back(mat.clone());
		}
		setParam();
	}

	// 5. 类移动构造 (零拷贝)
	// 用法: LFData new_lf = std::move(old_lf);
	LFData(LFData &&src) noexcept {
		data = std::move(src.data);
		setParam();
		// src 变成空，处于有效状态
		src.resetParam();
	}

	// 6. 移动赋值运算符
	LFData &operator=(LFData &&src) noexcept {
		if (this != &src) {
			data = std::move(src.data);
			setParam();
			src.resetParam();
		}
		return *this;
	}

	~LFData() {}

	cv::Mat &getSAI(int row, int col) {
		int idx = row * rows + col;
		if (idx < 0 || idx >= static_cast<int>(data.size())) {
			throw std::out_of_range("LFData::getSAI index out of range");
		}
		return data[idx]; // 返回引用，无拷贝
	}

	const cv::Mat &getSAI(int row, int col) const {
		int idx = row * rows + col;
		if (idx < 0 || idx >= static_cast<int>(data.size())) {
			throw std::out_of_range("LFData::getSAI index out of range");
		}
		return data[idx]; // 返回常引用，无拷贝
	}

	// 优化：增加 const 版本
	const cv::Mat &getCenter() const {
		if (data.empty())
			throw std::runtime_error("LFData is empty");
		return data[(size - 1) / 2];
	}

	// 保持非 const 版本供修改
	cv::Mat &getCenter() {
		if (data.empty())
			throw std::runtime_error("LFData is empty");
		return data[(size - 1) / 2];
	}

	bool empty() const { return data.empty(); }

	void clear() {
		data.clear();
		resetParam();
	}

	// 将参数更新逻辑独立
	void setParam() {
		if (data.empty()) {
			resetParam();
			return;
		}
		size = static_cast<int>(data.size());
		rows = static_cast<int>(std::sqrt(size));
		cols = rows;
		height = data[0].rows;
		width = data[0].cols;
		channels = data[0].channels();
		type = data[0].type();
	}

private:
	void resetParam() {
		size = rows = cols = height = width = channels = type = 0;
	}

	// 核心校验逻辑
	void validate(const std::vector<cv::Mat> &src) const {
		if (src.empty()) {
			// 允许空初始化，但不允许错误的数据
			return;
		}

		// 1. 检查是否为空图
		if (src[0].empty()) {
			throw std::invalid_argument("Input images must not be empty");
		}

		// 2. 检查是否为平方数 (光场必须是 N*N)
		int sz = static_cast<int>(src.size());
		int r = static_cast<int>(std::sqrt(sz));
		if (r * r != sz) {
			throw std::invalid_argument(
				"LightField size must be a perfect square (e.g. 9, 25, 81)");
		}

		// 3. (可选) 检查所有图片尺寸是否一致，防止 vector 里混入不同尺寸的图
		int h = src[0].rows;
		int w = src[0].cols;
		// 只检查最后一张作为抽样，或者检查全部
		if (src.back().rows != h || src.back().cols != w) {
			throw std::invalid_argument(
				"All sub-aperture images must have the same dimensions");
		}
	}
};

#endif