#ifndef LFIO_H
#define LFIO_H

#include "json.hpp"
#include "lfdata.h"

#include <opencv2/core.hpp>
#include <string>

using json = nlohmann::json;

class LFIO {
public:
	explicit LFIO();

	static cv::Mat read_image(const std::string &path);
	static LfPtr read_sai(const std::string &path, bool isRGB);
	LfPtr read_sai_openmp(const std::string &path, bool isRGB);

	static cv::Mat gamma_convert(const cv::Mat &src, bool inverse);

	// 保存所有视角的 Map
	// maps: 顺序存储 [View0_X, View0_Y, View1_X, View1_Y, ...]
	static bool saveLookUpTables(const std::string &path,
								 const std::vector<cv::Mat> &maps, int winSize);

	// 读取所有视角的 Map
	static bool loadLookUpTables(const std::string &path,
								 std::vector<cv::Mat> &maps, int &outWinSize);
};
#endif
