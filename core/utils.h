#ifndef UTILS_H
#define UTILS_H

#include "lfparams.h"

#include <chrono>
#include <json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using json = nlohmann::json;

class Timer {
public:
	using Clock = std::chrono::high_resolution_clock;
	using TimePoint = Clock::time_point;
	using Duration = Clock::duration;

	Timer();

	void start();

	Duration stop();
	Duration elapsed() const;
	double elapsed_s() const;
	double elapsed_ms() const;
	void print_elapsed_ms(const std::string & = "");

private:
	TimePoint _start_time; // 计时器开始的时间点
	Duration _duration;	   // 存储停止时的持续时间
	bool _is_running;	   // 计时器是否正在运行
};

cv::Mat draw_points(const cv::Mat &img,
					const std::vector<cv::Point2f> &pts_sorted,
					const std::string &output_path, int radius = 1,
					const cv::Scalar &color = cv::Scalar(0), bool save = false);
cv::Mat draw_points(const cv::Mat &img,
					const std::vector<cv::Point> &pts_sorted,
					const std::string &output_path, int radius = 1,
					const cv::Scalar &color = cv::Scalar(0), bool save = false);
cv::Mat draw_points(const cv::Mat &img,
					const std::vector<std::vector<cv::Point2f>> &pts_sorted,
					const std::string &output_path, int radius = 1,
					const cv::Scalar &color = cv::Scalar(0), bool save = false);
cv::Mat draw_points(const cv::Mat &img,
					const std::vector<std::vector<cv::Point>> &pts_sorted,
					const std::string &output_path, int radius = 1,
					const cv::Scalar &color = cv::Scalar(0), bool save = false);

json readJson(const std::string &path);
void writeJson(const std::string &path, const json &j, int indent = 4);
void saveAs8Bit(const std::string &path, const cv::Mat &img,
				double input_max_val = 1023.0);

void imshowRaw(const std::string &winname, const cv::Mat &img,
			   float resize_factor = 0.0f);
std::string get_base_filename(const std::string &filename);
cv::Mat gamma_convert(const cv::Mat &src, bool inverse);
int get_demosaic_code(BayerPattern pattern, bool gray = false);
#endif