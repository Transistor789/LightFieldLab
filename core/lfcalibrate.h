#ifndef LFCALIBRATE_H
#define LFCALIBRATE_H

#include <json.hpp>
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
	explicit LFCalibrate();
	explicit LFCalibrate(const cv::Mat &white_img);
	std::vector<std::vector<cv::Point2f>> run(bool use_cca = false,
											  bool save = false);
	void savePoints(const std::string &filename);

private:
	cv::Mat _white_img;
	std::vector<std::vector<cv::Point2f>> _points;
};

#endif