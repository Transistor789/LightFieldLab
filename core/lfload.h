#ifndef LFLOAD_H
#define LFLOAD_H

#include "config.h"
#include "json.hpp"
#include "lfdata.h"
#include "raw_decode.h"

#include <opencv2/core.hpp>
#include <string>

using json = nlohmann::json;

class LFLoad {
public:
	explicit LFLoad();

	cv::Mat read_image(const std::string &path);
	LfPtr read_sai(const std::string &path, bool isRGB);

	cv::Mat gamma_convert(const cv::Mat &src, bool inverse);

	LfPtr lfptr;

	json json_dict;
};
#endif
