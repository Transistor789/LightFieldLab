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

	void computeSliceMaps(int winSize);
	void computeSliceMaps(int winSize, cv::Mat &out_x, cv::Mat &out_y,
						  int row = -1, int col = -1);
	void getSliceMaps(cv::Mat &out_x, cv::Mat &out_y, int row = -1,
					  int col = -1);
	std::vector<cv::Mat> getSliceMaps() const { return _slice_maps; }

	void computeDehexMaps();
	void computeDehexMaps(cv::Mat &out_x, cv::Mat &out_y);
	void getDehexMaps(cv::Mat &out_x, cv::Mat &out_y);
	std::vector<cv::Mat> getDehexMaps() const { return _dehex_maps; }

private:
	cv::Mat _white_img;
	std::vector<std::vector<cv::Point2f>> _points;

	std::vector<cv::Mat> _slice_maps;
	std::vector<cv::Mat> _dehex_maps;

	void _computeSliceMaps(int winSize);
	void _computeDehexMaps();
};

#endif