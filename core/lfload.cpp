#include "lfload.h"

#include "config.h"

#include <filesystem>
#include <future>
#include <opencv2/opencv.hpp>
#include <openssl/sha.h>
#include <string>

LFLoad::LFLoad() {}

cv::Mat LFLoad::read_image(const std::string &path) {
	cv::Mat img;
	std::filesystem::path fs_path(path);
	std::string ext = fs_path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp"
		|| ext == ".tif" || ext == ".tiff") {
		img = cv::imread(path, cv::IMREAD_UNCHANGED);
		img = gamma_convert(img, true);
	} else {
		RawDecoder lfpdecoder;
		img = lfpdecoder.decode(path);
		// ISP::blc<uint16_t>(img);
		// ret = lfpdecoder.normalize_raw(ret);
		json_dict = lfpdecoder.json_dict;
	}
	return img;
}

cv::Mat LFLoad::gamma_convert(const cv::Mat &src, bool inverse) {
	if (src.empty()) {
		throw std::runtime_error("gammaConvert: src is empty!");
	}

	// 转换为 float32 格式，归一化到 [0, 1]
	cv::Mat srcFloat;
	if (src.depth() != CV_32F) {
		src.convertTo(srcFloat, CV_32F, 1.0 / 255.0);
	} else {
		srcFloat = src;
	}

	std::vector<cv::Mat> channels;
	cv::split(srcFloat, channels); // 拆分通道

	constexpr float A = 0.055f;
	constexpr float ALPHA = 1.055f;

	for (auto &ch : channels) {
		if (!inverse) {
			// 线性 -> sRGB
			constexpr float BETA = 0.0031308f;
			cv::Mat mask = ch >= BETA;

			cv::Mat gammaPart, linearPart;
			cv::pow(ch, 1.0f / 2.4f, gammaPart);
			gammaPart = ALPHA * gammaPart - A;

			linearPart = ch * 12.92f;
			gammaPart.copyTo(ch, mask);	  // 用 gamma 区覆盖
			linearPart.copyTo(ch, ~mask); // 用线性区覆盖其余部分
		} else {
			// sRGB -> 线性
			constexpr float THRESHOLD = 0.04045f;
			cv::Mat mask = ch > THRESHOLD;

			cv::Mat srcAdjusted = (ch + A) / ALPHA;
			cv::Mat gammaPart;
			cv::pow(srcAdjusted, 2.4f, gammaPart);

			cv::Mat linearPart = ch / 12.92f;
			gammaPart.copyTo(ch, mask);
			linearPart.copyTo(ch, ~mask);
		}
	}

	cv::Mat result;
	cv::merge(channels, result); // 合并通道
	return result;
}

LfPtr LFLoad::read_sai(const std::string &path, bool isRGB) {
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("read_sai: file not exist! Path: " + path);
	}

	auto start = std::chrono::high_resolution_clock::now();

	// 获取所有文件名
	std::vector<std::string> filenames;
	for (const auto &entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(entry.path().filename().string());
		}
	}
	std::sort(filenames.begin(), filenames.end());

	std::vector<cv::Mat> temp(filenames.size());
	std::vector<std::future<cv::Mat>> futures;

	for (const auto &i : filenames) {
		std::string filename = path + "/" + i;
		futures.push_back(
			std::async(std::launch::async, [&, isRGB, filename]() {
				return cv::imread(filename,
								  isRGB ? cv::IMREAD_COLOR
										: cv::IMREAD_GRAYSCALE); // 写入对应位置
			}));
	}

	for (int i = 0; i < futures.size(); ++i) {
		temp[i] = futures[i].get();
		temp[i].convertTo(temp[i], CV_32FC(temp[i].channels()), 1.0 / 255.0);
	}

	std::cout << "Loading finished!" << std::endl;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

	return std::make_shared<LFData>(std::move(temp));
}
