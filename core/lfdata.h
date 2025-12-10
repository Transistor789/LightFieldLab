#ifndef LFDATA_H
#define LFDATA_H

#include <opencv2/opencv.hpp>

enum { GRAY = 0, RGB = 1 };
enum { CPU = 0, GPU = 1 };
enum { RAW8 = 8, RAW10 = 10, RAW12 = 12, RAW16 = 16 };

class LFData {
public:
	explicit LFData() = default;
	explicit LFData(const std::vector<cv::Mat> &src) {
		data.reserve(src.size());
		for (const auto &mat : src) data.push_back(mat.clone());
		setParam();
	}
	explicit LFData(std::vector<cv::Mat> &&src) {
		data = std::move(src);
		setParam();
	}
	LFData(const LFData &src) {
		data.reserve(src.data.size());
		for (const auto &mat : src.data) {
			data.emplace_back(mat.clone()); // 深拷贝每个 cv::Mat
		}
		setParam();
	}
	~LFData() {}

	void toFloat() {
		if (type == CV_32FC(channels)) {
			return;
		}
		for (int i = 0; i < size; i++) {
			data[i].convertTo(data[i], CV_32FC(channels));
		}
	}

	void toUchar() {
		if (type == CV_8UC(channels)) {
			return;
		}
		for (int i = 0; i < size; i++) {
			cv::Mat temp;
			data[i].convertTo(data[i], CV_8UC(channels));
		}
	}
	cv::Mat getSAI(int row, int col, bool isUINT = false) {
		if (isUINT) {
			return data[row * rows + col].clone();
		}
		cv::Mat mat;
		data[row * rows + col].convertTo(mat, CV_8UC(channels), 255.0);
		return mat;
	}
	cv::Mat getCenter() const { return data[(1 + size) / 2 - 1]; }
	bool empty() const { return data.empty(); }
	void clear() { data.clear(); }

	void setParam() {
		size = data.size();
		rows = static_cast<int>(std::sqrt(data.size()));
		cols = rows;
		height = data[0].rows;
		width = data[0].cols;
		channels = data[0].channels();
		type = data[0].type();
	}

private:
	void validate(const std::vector<cv::Mat> &src) {
		if (src.empty()) {
			throw std::invalid_argument(
				"LightField requires a non-empty vector");
		}
		int size = src.size();
		int rows = static_cast<int>(std::sqrt(size));
		if (rows * rows != size) {
			throw std::invalid_argument(
				"LightField size must be a perfect square");
		}
		if (src[0].empty()) {
			throw std::invalid_argument("Input images must not be empty");
		}
	}
	void setParam(const std::vector<cv::Mat> &src) {
		size = src.size();
		rows = static_cast<int>(std::sqrt(src.size()));
		cols = rows;
		height = src[0].rows;
		width = src[0].cols;
		channels = src[0].channels();
		type = src[0].type();
	}

public:
	int size = 0;
	int rows = 0;
	int cols = 0;
	int height = 0;
	int width = 0;
	int channels = 0;
	int type = 0;

	std::vector<cv::Mat> data;
};
using LfPtr = std::shared_ptr<LFData>;

#endif