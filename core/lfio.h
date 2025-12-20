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
	static LfPtr read_sai(const std::string &path);

	static bool saveLookUpTables(const std::string &path,
								 const std::vector<cv::Mat> &maps, int winSize);

	static bool loadLookUpTables(const std::string &path,
								 std::vector<cv::Mat> &maps, int &outWinSize);
};
#endif
