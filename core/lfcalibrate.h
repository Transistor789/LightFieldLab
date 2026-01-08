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
		bool autoEstimate = true;
		int diameter = 0;
		int bitDepth = 8;
		BayerPattern bayer = BayerPattern::NONE;
		CentroidsExtract::Method ceMethod = CentroidsExtract::Method::Contour;
	} config;

	// 显式构造函数
	explicit LFCalibrate() = default;
	explicit LFCalibrate(const cv::Mat &white_img);

	// 设置标定图像
	void setImage(const cv::Mat &img);

	// 核心运行流程：检测、排序、拟合
	std::vector<std::vector<cv::Point2f>> run();

	// 工具函数
	void initConfigLytro2();
	void savePoints(const std::string &filename);

	// 获取检测到的奇偶行相位 (用于调试或外部验证)
	bool getHexOdd() const { return _hex_odd; }

	// --- 映射表计算 (Map Generation) ---

	// 计算并缓存 Extract Maps (用于从光场图提取微透镜图像)
	// 返回引用避免拷贝
	const std::vector<cv::Mat> &computeExtractMaps(int winSize);

	// 获取指定视角的 Extract Map
	void getExtractMaps(cv::Mat &out_x, cv::Mat &out_y, int row = -1,
						int col = -1) const;

	// 计算并缓存 Dehex Maps (用于将六边形采样转为矩形)
	const std::vector<cv::Mat> &computeDehexMaps();

	// 获取 Dehex Maps
	void getDehexMaps(cv::Mat &out_x, cv::Mat &out_y);

private:
	// 内部 Worker 函数
	void _computeExtractMaps(int winSize);
	void _computeDehexMaps();

private:
	cv::Mat _white_img;
	std::vector<std::vector<cv::Point2f>> _points;

	// 标定状态
	bool _hex_odd = false; // [新增] 存储从排序步骤获取的奇偶行相位

	// 缓存
	std::vector<cv::Mat> _extract_maps;
	std::vector<cv::Mat> _dehex_maps;
};

#endif // LFCALIBRATE_H