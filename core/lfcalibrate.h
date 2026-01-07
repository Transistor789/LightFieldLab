#ifndef LFCALIBRATE_H
#define LFCALIBRATE_H

#include "json.hpp"
#include "utils.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace nlohmann {
template <>
struct adl_serializer<cv::Point2f> {
	static void to_json(json &j, const cv::Point2f &p) { j = {p.x, p.y}; }
	static void from_json(const json &j, cv::Point2f &p) {
		p.x = j.at(0).get<float>();
		p.y = j.at(1).get<float>();
	}
};
} // namespace nlohmann

class LFCalibrate {
public:
	struct Config {
		bool autoEstimate = true;
		bool use_cca = false;
		int diameter = 0;
		int bitDepth = 8;
		BayerPattern bayer = BayerPattern::NONE;
	} config;

	explicit LFCalibrate() = default;
	explicit LFCalibrate(const cv::Mat &white_img);
	std::vector<std::vector<cv::Point2f>> run();
	void initConfigLytro2();
	void setImage(const cv::Mat &img);
	void savePoints(const std::string &filename);

	std::vector<cv::Mat> computeExtractMaps(int winSize);
	void computeExtractMaps(int winSize, cv::Mat &out_x, cv::Mat &out_y,
							int row = -1, int col = -1);
	void getExtractMaps(cv::Mat &out_x, cv::Mat &out_y, int row = -1,
						int col = -1);
	std::vector<cv::Mat> getExtractMaps() const { return _extract_maps; }

	std::vector<cv::Mat> computeDehexMaps();
	void computeDehexMaps(cv::Mat &out_x, cv::Mat &out_y);
	void getDehexMaps(cv::Mat &out_x, cv::Mat &out_y);
	std::vector<cv::Mat> getDehexMaps() const { return _dehex_maps; }

private:
	void _computeExtractMaps(int winSize);
	void _computeDehexMaps();

private:
	cv::Mat _white_img;
	std::vector<std::vector<cv::Point2f>> _points;

	std::vector<cv::Mat> _extract_maps;
	std::vector<cv::Mat> _dehex_maps;
};

#endif