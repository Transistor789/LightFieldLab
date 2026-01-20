#ifndef LFCALIBRATE_H
#define LFCALIBRATE_H

#include "centers_extract.h"
#include "hexgrid_fit.h"
#include "json.hpp"
#include "utils.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <utility>
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

enum class Orientation { HORZ, VERT };

struct CalibrateConfig {
	bool hexgridfit = false;
	bool genLUT = false;
	bool autoEstimate = false;
	int diameter = 0;
	int bitDepth = 8;
	int views = 9;
	float space = 1.0;
	BayerPattern bayer = BayerPattern::NONE;
	ExtractMethod ceMethod = ExtractMethod::Contour;
	Orientation orientation = Orientation::HORZ;
};

class LFCalibrate {
public:
	// 显式构造函数
	explicit LFCalibrate() = default;
	explicit LFCalibrate(const cv::Mat &white_img);

	// 设置标定图像
	void setImage(const cv::Mat &img);

	// 核心运行流程：检测、排序、拟合
	void run(const cv::Mat &img, const CalibrateConfig &cfg);

	// 工具函数
	void savePoints(const std::string &filename);

	bool getHexOdd() const { return _hex_odd; }
	int getDiameter() const { return _diameter; }
	std::vector<cv::Mat> getExtractMaps() const;
	std::vector<cv::Mat> getDehexMaps() const;
	std::shared_ptr<HexGridFitter> getFitter() const { return _fitter; }
	bool isExtractLutEmpty() const { return _extract_maps.empty(); }
	bool isDehexLutEmpty() const { return _dehex_maps.empty(); }

	std::pair<cv::Mat, cv::Mat> getPoints() const { return _maps; }

	const std::vector<cv::Mat> &computeExtractMaps(int winSize, float space = 1.0f);
	const std::vector<cv::Mat> &computeDehexMaps();

	// 获取 Dehex Maps

private:
	cv::Mat _white_img;
	std::pair<cv::Mat, cv::Mat> _maps;

	// 标定状态
	bool _hex_odd = false; // [新增] 存储从排序步骤获取的奇偶行相位
	int _diameter = 0;

	// 缓存
	std::vector<cv::Mat> _extract_maps;
	std::vector<cv::Mat> _dehex_maps;
	std::shared_ptr<HexGridFitter> _fitter = nullptr;
};

#endif // LFCALIBRATE_H