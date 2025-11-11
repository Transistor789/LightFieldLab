#ifndef LFLOAD_H
#define LFLOAD_H

#include "json.hpp"
#include "lfdata.h"

#include <string>

using json = nlohmann::json;
constexpr static int LYTRO_WIDTH = 7728;
constexpr static int LYTRO_HEIGHT = 5368;
constexpr static int LYTRO_BUFFER_INDEX = 10;
constexpr static int LYTRO_BAYER = cv::COLOR_BayerGR2RGB;

class LFLoad {
public:
	LFLoad() = default;

	static cv::Mat convertRaw8ToMat(uint8_t *src, int width, int height,
									bool isFloat = true);
	static cv::Mat convertRaw10ToMat(const uint8_t *src, int width, int height,
									 bool isFloat = true);
	static cv::Mat gammaConvert(const cv::Mat &src, bool inverse);
	static LightFieldPtr loadSAI(const std::string &path, bool isRGB);
	static int loadRaw(const std::string &filename, cv::Mat &dst, int width,
					   int height, int bitDepth);
	static std::vector<uint8_t> readRawFile(const std::string &filename);
	static json readJson(const std::vector<std::string> &sections);
	static std::string readSection(std::ifstream &file);
	static std::vector<std::string> readLytroFile(const std::string &path);
	static cv::Mat loadImageFile(const std::string &filename,
								 int width = LYTRO_WIDTH,
								 int height = LYTRO_HEIGHT);
	static bool hasExtension(const std::string &filename,
							 const std::string &ext);
};

#endif