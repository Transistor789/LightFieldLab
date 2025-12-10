#include "utils.h"

#include <fstream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <ratio>
#include <string>

Timer::Timer() : _is_running(false) { start(); }

void Timer::start() {
	_start_time = std::chrono::high_resolution_clock::now();
	_is_running = true;
	_duration = Duration::zero(); // 重置存储的持续时间
}

Timer::Duration Timer::stop() {
	if (!_is_running) {
		return _duration;
	}

	TimePoint end_time = std::chrono::high_resolution_clock::now();
	_duration = end_time - _start_time;
	_is_running = false;
	return _duration;
}

Timer::Duration Timer::elapsed() const {
	if (_is_running) {
		return std::chrono::high_resolution_clock::now() - _start_time;
	}
	return _duration;
}

double Timer::elapsed_s() const {
	return std::chrono::duration<double>(elapsed()).count();
}
double Timer::elapsed_ms() const {
	return std::chrono::duration<double, std::milli>(elapsed()).count();
}
void Timer::print_elapsed_ms(const std::string &message) {
	if (message.empty()) {
		printf("耗时 %.3f ms\n", elapsed_ms());
	} else {
		printf("%s 耗时 %.3f ms\n", message.c_str(), elapsed_ms());
	}
}

cv::Mat draw_points(const cv::Mat &img, const std::vector<cv::Point2f> &points,
					const std::string &output_path, int radius,
					const cv::Scalar &color, bool save) {
	cv::Mat temp = img.clone();
	for (const auto &pt : points) {
		cv::circle(temp, pt, radius, color, -1);
	}
	if (save) {
		cv::imwrite(output_path, temp);
	}
	return temp;
}
cv::Mat draw_points(const cv::Mat &img,
					const std::vector<std::vector<cv::Point2f>> &points,
					const std::string &output_path, int radius,
					const cv::Scalar &color, bool save) {
	cv::Mat temp = img.clone();
	for (const std::vector<cv::Point2f> &row : points) {
		for (const auto &pt : row) {
			cv::circle(temp, pt, radius, color, -1);
		}
	}
	if (save) {
		cv::imwrite(output_path, temp);
	}
	return temp;
}

/**
 * @brief 读取 JSON 文件 (健壮版)
 * @param path 文件路径
 * @return json 对象
 * @throws std::runtime_error 如果文件无法打开或解析失败
 */
json readJson(const std::string &path) {
	// 1. 尝试打开文件
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Error: Unable to open file for reading: "
								 + path);
	}

	// 2. 检查文件是否为空（可选，但推荐）
	if (file.peek() == std::ifstream::traits_type::eof()) {
		throw std::runtime_error("Error: File is empty: " + path);
	}

	try {
		// 3. 解析 JSON
		json j;
		file >> j; // 或者 j = json::parse(file);
		return j;
	} catch (const json::parse_error &e) {
		// 4. 捕获特定的 JSON 语法错误
		std::string err =
			"Error: JSON parsing failed in " + path + "\nMessage: " + e.what();
		throw std::runtime_error(err);
	}
}

/**
 * @brief 写入 JSON 文件 (健壮版)
 * @param path 文件路径
 * @param j JSON 对象
 * @param indent 缩进空格数，默认为 4 (美化输出)，设为 -1 则为紧凑模式
 * @throws std::runtime_error 如果文件无法写入
 */
void writeJson(const std::string &path, const json &j, int indent) {
	// 1. 尝试打开文件
	std::ofstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Error: Unable to open file for writing: "
								 + path);
	}

	// 2. 写入数据
	// indent 参数用于控制缩进，-1 为紧凑格式，4 为标准美化格式
	file << j.dump(indent);

	// 3. 检查写入状态（防止磁盘满等情况）
	if (file.bad()) {
		throw std::runtime_error("Error: Failed to write data to file: "
								 + path);
	}

	// file 析构时会自动 close，但手动检查 bad bit 是好习惯
}

/**
 * @brief 将高位深图像归一化并保存为 8-bit 图像 (用于预览/可视化)
 * @param path 保存路径
 * @param img 输入图像 (可以是 CV_16U, CV_32F 等)
 * @param input_max_val 输入图像的白点值 (10-bit=1023, 12-bit=4095, Float=1.0)
 * 默认为 1023.0 (适配你的场景)
 */
void saveAs8Bit(const std::string &path, const cv::Mat &img,
				double input_max_val) {
	// 1. 鲁棒性检查：空图像
	if (img.empty()) {
		throw std::invalid_argument("Input image is empty, cannot save to "
									+ path);
	}

	// 2. 鲁棒性检查：确保目录存在 (C++17)
	std::filesystem::path fs_path(path);
	if (fs_path.has_parent_path()) {
		std::filesystem::create_directories(fs_path.parent_path());
	}

	// 3. 转换逻辑
	cv::Mat img_u8;
	// 计算缩放因子: 目标最大值(255) / 输入最大值
	double scale_factor = 255.0 / input_max_val;

	// convertTo 会自动处理饱和度截断 (Saturate Cast)，防止溢出
	// 第三个参数 alpha 是缩放系数
	img.convertTo(img_u8, CV_8U, scale_factor);

	// 4. 写入检查
	bool success = cv::imwrite(path, img_u8);
	if (!success) {
		throw std::runtime_error("Failed to write image file to: " + path);
	}
}

std::string get_base_filename(const std::string &filename) {
	// 1. 查找最后一个点号 '.' 的位置
	size_t last_dot_pos = filename.find_last_of('.');

	// 2. 检查是否找到点号
	// 如果找不到点号，或者点号是第一个字符（隐藏文件，如
	// .bashrc），则返回原字符串
	if (last_dot_pos == std::string::npos || last_dot_pos == 0) {
		return filename;
	}

	// 3. 使用 substr 截取从开始到点号前的部分
	return filename.substr(0, last_dot_pos);
}