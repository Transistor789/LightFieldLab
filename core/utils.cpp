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

int get_demosaic_code(BayerPattern pattern, bool gray) {
	switch (pattern) {
		case BayerPattern::GRBG:
			return gray ? cv::COLOR_BayerGR2GRAY
						: cv::COLOR_BayerGR2RGB; // 第0行是 G, R
		case BayerPattern::RGGB:
			return gray ? cv::COLOR_BayerRG2GRAY
						: cv::COLOR_BayerRG2RGB; // 第0行是 R, G
		case BayerPattern::GBRG:
			return gray ? cv::COLOR_BayerGB2GRAY
						: cv::COLOR_BayerGB2RGB; // 第0行是 G, B
		case BayerPattern::BGGR:
			return gray ? cv::COLOR_BayerBG2GRAY
						: cv::COLOR_BayerBG2RGB; // 第0行是 B, G
		default:
			// 默认处理
			return gray ? cv::COLOR_BayerGR2GRAY : cv::COLOR_BayerGR2RGB;
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
cv::Mat draw_points(const cv::Mat &img, const std::vector<cv::Point> &points,
					const std::string &output_path, int radius,
					const cv::Scalar &color, bool save) {
	cv::Mat temp = img.clone();
	for (const auto &pt : points) {
		cv::circle(temp, pt, radius, color, 1);
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
cv::Mat draw_points(const cv::Mat &img,
					const std::vector<std::vector<cv::Point>> &points,
					const std::string &output_path, int radius,
					const cv::Scalar &color, bool save) {
	cv::Mat temp = img.clone();
	for (const std::vector<cv::Point> &row : points) {
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

void imshowRaw(const std::string &winname, const cv::Mat &img,
			   float resize_factor) {
	// 1. 鲁棒性检查
	if (img.empty()) {
		std::cerr << "[Warning] imshowRaw: Image is empty!" << std::endl;
		return;
	}

	cv::Mat display_img;

	// 2. 自动类型推断与归一化
	int depth = img.depth();

	if (depth == CV_8U) {
		// 如果已经是 8-bit，直接使用
		display_img = img;
	} else if (depth == CV_16U || depth == CV_16S || depth == CV_32F
			   || depth == CV_64F) {
		// 对于 16-bit Raw 或 浮点图，执行 MinMax 归一化
		// 这样无论你是 10-bit (1023) 还是 12-bit (4095)，
		// 都会被自动拉伸到 0-255，显示效果最好（对比度最高）
		cv::normalize(img, display_img, 0, 255, cv::NORM_MINMAX, CV_8U);

		// 备选方案：如果你希望保持绝对亮度（比如不想让全黑的图变亮），可以用固定缩放：
		// double scale = 255.0 / 65535.0; // 假设占满16位
		// img.convertTo(display_img, CV_8U, scale);
	} else {
		std::cerr << "[Warning] imshowRaw: Unsupported depth!" << std::endl;
		return;
	}

	// 3. 缩放显示 (防止撑爆屏幕)
	if (resize_factor > 0.0f && std::abs(resize_factor - 1.0f) > 1e-5) {
		// 如果原图是 Bayer 格式 (单通道)，INTER_NEAREST
		// 可以保留马赛克特征以便观察 如果是 RGB 图，INTER_AREA 更好
		int interpolation =
			(img.channels() == 1) ? cv::INTER_NEAREST : cv::INTER_AREA;
		cv::resize(display_img, display_img, cv::Size(), resize_factor,
				   resize_factor, interpolation);
	}

	// 4. 显示
	cv::imshow(winname, display_img);
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

cv::Mat gamma_convert(const cv::Mat &src, bool inverse) {
	if (src.empty())
		throw std::runtime_error("Src empty");

	cv::Mat result;
	// 确保输入是 float
	if (src.depth() != CV_32F) {
		src.convertTo(result, CV_32F, 1.0 / 255.0);
	} else {
		result = src.clone(); // 深拷贝，避免修改原图
	}

	int total_pixels = result.rows * result.cols * result.channels();

	// 获取连续内存指针（如果连续）
	if (result.isContinuous()) {
		total_pixels = total_pixels; // 数量不变
		// reshape 成 1行，方便并行
		result = result.reshape(1, 1);
	}

	float *ptr = result.ptr<float>();

	constexpr float A = 0.055f;
	constexpr float ALPHA = 1.055f;
	constexpr float BETA = 0.0031308f;
	constexpr float THRESHOLD = 0.04045f;
	constexpr float INV_GAMMA = 1.0f / 2.4f;
	constexpr float GAMMA = 2.4f;

// 并行处理所有像素
#pragma omp parallel for
	for (int i = 0; i < total_pixels; ++i) {
		float v = ptr[i];
		// 简单的 clamp 防止负数或溢出（可选）
		if (v < 0.0f)
			v = 0.0f;
		else if (v > 1.0f)
			v = 1.0f;

		if (!inverse) {
			// Linear -> sRGB
			if (v <= BETA) {
				ptr[i] = v * 12.92f;
			} else {
				// std::pow 比较慢，但在 float 精度下难以避免，
				// 除非用 SSE/AVX 近似指令
				ptr[i] = ALPHA * std::pow(v, INV_GAMMA) - A;
			}
		} else {
			// sRGB -> Linear
			if (v <= THRESHOLD) {
				ptr[i] = v / 12.92f;
			} else {
				ptr[i] = std::pow((v + A) / ALPHA, GAMMA);
			}
		}
	}

	// 如果之前 reshape 过，OpenCV 会自动处理维度，
	// 但如果 result 是新创建的 reshape 后的 Mat，这里需要 reshape 回去。
	// 不过由于我们 clone 并在原数据上操作，只要 reshape 只是改头信息，
	// 最好在 return 前 reshape 回原始尺寸：
	if (src.rows != result.rows || src.cols != result.cols)
		return result.reshape(src.channels(), src.rows);

	return result;
}

ImageFileType checkImageType(const std::string &path) {
	namespace fs = std::filesystem;
	fs::path p(path);

	// 如果没有后缀，默认当做普通图像处理，或者您可以定义一个 Unknown
	if (!p.has_extension())
		return ImageFileType::Normal;

	// 1. 获取后缀
	std::string ext = p.extension().string();

	// 2. 统一转为小写 (这样就自动覆盖了 .RAW, .LFP, .Lfr 等情况)
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	// 3. 分类判断
	if (ext == ".lfp" || ext == ".lfr") {
		return ImageFileType::Lytro;
	}

	if (ext == ".raw") {
		return ImageFileType::Raw;
	}

	// 其他所有情况 (png, jpg, tif, bmp, etc.)
	return ImageFileType::Normal;
}