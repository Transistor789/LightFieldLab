#ifndef LFCALIBRATE_H
#define LFCALIBRATE_H

#include "centers_extract.h"
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
		bool genLUT = true;
		bool saveLUT = false;
		bool autoEstimate = true;
		int diameter = 0;
		int bitDepth = 8;
		int views = 9;
		BayerPattern bayer = BayerPattern::NONE;
		CentroidsExtract::Method ceMethod = CentroidsExtract::Method::Contour;
	};

	// 显式构造函数
	explicit LFCalibrate() = default;
	explicit LFCalibrate(const cv::Mat &white_img);

	// 设置标定图像
	void setImage(const cv::Mat &img);

	// 核心运行流程：检测、排序、拟合
	void run(const LFCalibrate::Config &cfg);

	// 工具函数
	void savePoints(const std::string &filename);

	bool getHexOdd() const { return _hex_odd; }
	int getDiameter() const { return _diameter; }
	std::vector<cv::Mat> getExtractMaps() const;
	std::vector<cv::Mat> getDehexMaps() const;
	bool isExtractLutEmpty() const { return _extract_maps.empty(); }
	bool isDehexLutEmpty() const { return _dehex_maps.empty(); }

	std::vector<std::vector<cv::Point2f>> getPoints() const { return _points; }

	const std::vector<cv::Mat> &computeExtractMaps(int winSize);
	const std::vector<cv::Mat> &computeDehexMaps();

	// 获取 Dehex Maps

private:
	// 内部 Worker 函数
	void _computeExtractMaps(int winSize);
	void _computeDehexMaps();

private:
	cv::Mat _white_img;
	std::vector<std::vector<cv::Point2f>> _points;

	// 标定状态
	bool _hex_odd = false; // [新增] 存储从排序步骤获取的奇偶行相位
	int _diameter = 0;

	// 缓存
	std::vector<cv::Mat> _extract_maps;
	std::vector<cv::Mat> _dehex_maps;
};

#endif // LFCALIBRATE_H