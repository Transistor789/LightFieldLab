#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "lfcalibrate.h"
#include "lfio.h"
#include "utils.h"

#include <format>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

void test_module();
void test_calibrate();
void test_json();
void test_lut();
void test_detach();

int main() {
	// test_module();
	// test_calibrate();
	// test_json();
	// test_lut();
	test_detach();

	return 0;
}

void test_module() {
	cv::Mat img = cv::imread("../../data/gray.png", cv::IMREAD_GRAYSCALE);
	// cv::imshow("", img);
	// cv::waitKey();

	bool use_cca = false;

	Timer timer;

	CentroidsExtract ce(img);

	ce.run(ExtractMethod::Contour);
	timer.stop();

	std::vector<cv::Point2f> pts = ce.getPoints();
	std::vector<float> pitch = ce.getPitch();

	std::cout << "--- CentroidsExtract ---" << std::endl;
	std::cout << "检测到的点个数: " << pts.size() << std::endl;
	std::cout << "间距: " << pitch[0] << " " << pitch[1] << std::endl;
	std::cout << "耗时: " << timer.elapsed_ms() << " ms" << std::endl;

	CentroidsSort cs(pts, {pitch[0], pitch[1]});

	// test
	// cv::Point2f top_left = cs._top_left;
	// cv::Point2f bottom_right = cs._bottom_right;
	// std::cout << "当前点: " << top_left << std::endl;
	// std::cout << "右: " << cs.next_horz(top_left, false) << std::endl;
	// std::cout << "左: " << cs.next_horz(top_left, true) << std::endl;
	// std::cout << "左下: " << cs.next_vert(top_left, false, false) <<
	// std::endl; std::cout << "右下: " << cs.next_vert(top_left, false,
	// true)
	// << std::endl; std::cout << "左上: " << cs.next_vert(top_left, true,
	// true)
	// << std::endl; std::cout << "右上: " << cs.next_vert(top_left, true,
	// true)
	// << std::endl; auto [top_right, xsize] =
	// cs.count_points_horz(top_left, bottom_right, false); auto
	// [bottom_left, ysize] = 	cs.count_points_vert(top_left, bottom_right,
	// false, true); std::cout << "top_right: " << top_right << " xsize: "
	// << xsize << std::endl; std::cout << "bottom_left: " << bottom_left <<
	// " ysize: " << ysize
	// 		  << std::endl;

	timer.start();
	cs.run();
	timer.stop();
	std::vector<cv::Point2f> pts_sorted = cs.getPoints();
	std::vector<int> size = cs.getPointsSize();
	bool hex_odd = cs.getHexOdd();
	std::cout << "--- CentersSort ---" << std::endl;
	std::cout << "排列后的点个数: " << pts_sorted.size() << std::endl;
	std::cout << "微透镜尺寸: " << size[0] << ", " << size[1] << std::endl;
	std::cout << "hex_odd: " << hex_odd << std::endl;
	std::cout << "耗时: " << timer.elapsed_ms() << " ms" << std::endl;

	HexGridFitter hgf(cs.getPointsAsMats(), hex_odd);
	timer.start();
	hgf.fit();
	timer.stop();

	// 修复点 2: predict() 现在返回 std::pair<cv::Mat, cv::Mat>
	auto pts_fitted = hgf.predict();

	std::cout << "--- HexGridFitter ---" << std::endl;
	// 修复点 3: pair 不支持 size()，通过矩阵属性获取尺寸
	std::cout << "六角拟合后的网格尺寸: " << pts_fitted.first.rows << " x " << pts_fitted.first.cols << std::endl;
	std::cout << "耗时: " << timer.elapsed_ms() << " ms" << std::endl;

	auto info = hgf.get_grid_info();
	// 访问参数
	cv::Point2f origin = info.origin; // 起始点 (y, x)
	float pitch_row = info.pitch_row; // 行间距
	float pitch_col = info.pitch_col; // 列间距
	float rmse = info.rmse;			  // 拟合误差

	std::cout << "origin: " << origin << std::endl;
	std::cout << "pitch_row: " << pitch_row << std::endl;
	std::cout << "pitch_col: " << pitch_col << std::endl;
	std::cout << "rmse: " << rmse << std::endl;
}

void test_calibrate() {
	// cv::Mat img = cv::imread("../../data/gray.png", cv::IMREAD_GRAYSCALE);
	json meta;
	cv::Mat img =
		LFIO::ReadWhiteImageAuto("D:/code/LightFieldLab/data/toy.lfr", "D:/code/LightFieldCamera/B5152102610/", meta);
	imshowRaw("raw", img);
	cv::waitKey();
	// cv::Mat img = LFIO::ReadLFP("../../data/MOD_0015.RAW");
	// Timer timer;

	cv::Mat demosic;
	cv::demosaicing(img, demosic, cv::COLOR_BayerGB2GRAY);
	demosic.convertTo(demosic, CV_8U, 255.0 / 1023.0);
	imshowRaw("demosic", demosic);
	cv::imwrite("../../models/Demosaic.png", demosic);
	cv::waitKey();

	// LFCalibrate cali(img);
	// timer.start();
	// cali.config.use_cca = false;
	// cali.config.bayer = BayerPattern::GRBG;
	// cali.config.bitDepth = 10;
	// auto pts_cali = cali.run();
	// timer.stop();
	// std::cout << "--- Calibrate ---" << std::endl;
	// std::cout << "pts_cali size: " << pts_cali.size() << " "
	// 		  << pts_cali[0].size() << std::endl;
	// std::cout << "总耗时: " << timer.elapsed_ms() << " ms" << std::endl;
}

void test_json() {
	std::string path = "../../data/centroids.json";
	Timer timer;
	json j = readJson(path);
	timer.stop();
	timer.print_elapsed_ms();

	timer.start();
	writeJson(path, j);
	timer.stop();
	timer.print_elapsed_ms();

	json cal_data = readJson("../../data/MOD_0015.TXT");
	std::cout << cal_data.dump(4) << std::endl;
	writeJson("../../data/MOD_0015.json", cal_data);
}

void test_lut() {
	cv::Mat img = cv::imread("../../data/gray.png", cv::IMREAD_GRAYSCALE);
	LFCalibrate cali(img);
	Timer timer;

	CalibrateConfig config;
	config.ceMethod = ExtractMethod::Contour;
	config.bayer = BayerPattern::NONE;
	config.bitDepth = 8;
	cali.run(img, config);
	timer.stop();
	timer.print_elapsed_ms();

	for (int winSize = 1; winSize <= 13; winSize += 2) {
		LFIO::SaveLookUpTables(std::format("../../data/calibration/slice_{}.bin", winSize),
							   cali.computeExtractMaps(winSize), winSize);

		LFIO::SaveLookUpTables("../../data/calibration/lut_dehex.bin", cali.computeDehexMaps(), 1);
	}
}

void test_detach() {
	// 1. 读取 Raw 图
	// 请确保路径正确
	json j;
	cv::Mat img =
		LFIO::ReadWhiteImageAuto("D:/code/LightFieldLab/data/toy.lfr", "D:/code/LightFieldCamera/B5152102610/", j);

	if (img.empty()) {
		std::cerr << "Failed to load image!" << std::endl;
		return;
	}

	// 2. 简单的去马赛克转灰度
	cv::Mat gray;
	// 注意：请确认你的 Raw 是 GB 格式，如果这里错了，灰度图会有网格纹理
	cv::demosaicing(img, gray, cv::COLOR_BayerGB2GRAY);

	// 3. 转为 8-bit (DoG 在 8-bit 上够用了，且速度快)
	// 假设是 10-bit raw (0-1023)
	gray.convertTo(gray, CV_8U, 255.0 / 1023.0);
	imshowRaw("1. Original Gray", gray);

	// =========================================================
	// DoG 分离处理核心代码
	// =========================================================

	// [参数设置] 估计你的微透镜直径 (像素)
	// 如果微透镜很大，这里一定要改大！比如 30, 50, 80...
	double diameter = 15.0;

	// sigma1: 控制核心 (保留高频细节) -> 约直径的 1/6 ~ 1/4
	// sigma2: 控制背景 (抑制低频粘连) -> 约直径的 1/2 ~ 1
	double sigma1 = std::max(1.0, diameter * 0.2);
	double sigma2 = std::max(3.0, diameter * 0.8);

	std::cout << "Running DoG with sigma1=" << sigma1 << ", sigma2=" << sigma2 << std::endl;

	// A. CLAHE (限制对比度自适应直方图均衡)
	// 这一步非常重要！它能把灰蒙蒙的粘连处强行拉开反差
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
	cv::Mat enhanced;
	clahe->apply(gray, enhanced);
	imshowRaw("2. CLAHE Enhanced", enhanced);

	// B. 计算 DoG (Difference of Gaussians)
	cv::Mat g1, g2, dog;
	cv::GaussianBlur(enhanced, g1, cv::Size(0, 0), sigma1);
	cv::GaussianBlur(enhanced, g2, cv::Size(0, 0), sigma2);

	// 核心：中心 - 背景
	// 这样做的物理含义是：保留突出的亮点，减去平缓的背景
	cv::subtract(g1, g2, dog);

	// C. 归一化 (为了显示清晰)
	// 拉伸到 0-255
	cv::normalize(dog, dog, 0, 255, cv::NORM_MINMAX);
	imshowRaw("3. DoG Result (Separated)", dog);

	// (可选) 二值化看看效果
	cv::Mat binary;
	// 取前 20% 亮度的区域作为中心
	// DoG 后背景基本是黑的，阈值切分非常容易
	double minVal, maxVal;
	cv::minMaxLoc(dog, &minVal, &maxVal);
	cv::threshold(dog, binary, maxVal * 0.4, 255, cv::THRESH_BINARY);
	imshowRaw("4. Binary Result", binary);

	cv::imwrite("../data/gray.png", gray);
	cv::imwrite("../data/enhanced.png", enhanced);
	cv::imwrite("../models/Dog.png", dog);
	cv::imwrite("../data/binary.png", binary);

	cv::waitKey();
}