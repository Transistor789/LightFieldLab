#ifndef RAWDECODE_H
#define RAWDECODE_H

#include "json.hpp"

#include <fstream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>


using json = nlohmann::json;

class RawDecoder {
public:
	explicit RawDecoder() = default;

	/**
	 * @brief 解码 LFP/LFR 格式文件
	 * @param filename 文件路径
	 * @param outMetadata [输出] 解析出的关键元数据 (BLC, AWB, CCM 等)
	 * @return 解码后的 Bayer 图像 (CV_16UC1)
	 */
	cv::Mat DecodeLytro(const std::string &filename, json &outMetadata);

	/**
	 * @brief 解码纯 RAW 格式文件 (仅像素数据)
	 * @param filename 文件路径
	 * @return 解码后的 Bayer 图像 (CV_16UC1)
	 */
	cv::Mat DecodeRaw(const std::string &filename);

	/**
	 * @brief 解码白板图像 (RAW + TXT/JSON)
	 * 自动寻找同名元数据文件，并应用 BLC 和 AWB
	 * @param filename RAW 文件路径
	 * @param outMetadata [输出] 找到并使用的白图元数据
	 * @return 修正后的 16位 图像
	 */
	cv::Mat DecodeWhiteImage(const std::string &filename, json &outMetadata);

private:
	// === 内部工具函数 (全部为纯函数，无状态) ===

	// I/O 相关
	std::vector<std::string> ReadLytroFile(const std::string &filename);
	std::string ReadSection(std::ifstream &file);
	std::vector<uint8_t> ReadRawFile(const std::string &filename);

	// JSON 处理
	json ExtractJson(const std::vector<std::string> &sections);
	json FilterLfpJson(const json &jsonDict);
	json LoadWhiteMetadata(const std::string &rawFilename);

	// 图像算法
	cv::Mat UnpackRaw10ToBayer(const uint8_t *src, int width, int height);
	void ApplyWhiteBalance(cv::Mat &raw, const json &metadata);
};

#endif // RAWDECODE_H