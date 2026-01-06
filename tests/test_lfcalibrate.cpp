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

int main() {
	// test_module();
	test_calibrate();
	// test_json();
	// test_lut();

	return 0;
}

void test_module() {
	cv::Mat img = cv::imread("../../data/gray.png", cv::IMREAD_GRAYSCALE);
	// cv::imshow("", img);
	// cv::waitKey();

	bool use_cca = false, save = true;

	Timer timer;

	CentroidsExtract ce(img);

	ce.run(use_cca);
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

	HexGridFitter hgf(pts_sorted, size, hex_odd);
	timer.start();
	hgf.fit();
	timer.stop();
	auto pts_fitted = hgf.predict();
	std::cout << "--- HexGridFitter ---" << std::endl;
	std::cout << "六角拟合后的点个数: " << pts_sorted.size() << std::endl;
	std::cout << "六角拟合后的微透镜尺寸: " << pts_fitted.size() << " "
			  << pts_fitted[0].size() << std::endl;
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

	if (save) {
		if (use_cca) {
			draw_points(img, pts, "../../data/centers_detected_cca.png", 1, 0,
						true);
		} else {
			draw_points(img, pts, "../../data/centers_detected_moments.png", 1,
						0, true);
		}

		draw_points(img, pts_sorted, "../../data/centers_sorted.png", 1, 0,
					true);
		draw_points(img, pts_fitted, "../../data/centers_fitted.png", 1, 0,
					true);
	}
}

void test_calibrate() {
	// cv::Mat img = cv::imread("../../data/gray.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img = LFIO::readLFP("../../data/MOD_0015.RAW");
	Timer timer;

	// cv::Mat demosic;
	// cv::demosaicing(img, demosic, cv::COLOR_BayerGB2GRAY);
	// demosic.convertTo(demosic, CV_8U, 255.0 / 1023.0);

	LFCalibrate cali(img);
	timer.start();
	auto pts_cali = cali.run(false, true, 10);
	timer.stop();
	std::cout << "--- Calibrate ---" << std::endl;
	std::cout << "pts_cali size: " << pts_cali.size() << " "
			  << pts_cali[0].size() << std::endl;
	std::cout << "总耗时: " << timer.elapsed_ms() << " ms" << std::endl;
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
	auto pts_cali = cali.run(false, false, 8);
	timer.stop();
	timer.print_elapsed_ms();

	for (int winSize = 1; winSize <= 13; winSize += 2) {
		cali.computeSliceMaps(winSize);
		LFIO::saveLookUpTables(
			std::format("../../data/calibration/slice_{}.bin", winSize),
			cali.getSliceMaps(), winSize);
		cali.computeDehexMaps();
		LFIO::saveLookUpTables("../../data/calibration/dehex.bin",
							   cali.getDehexMaps(), 1);
	}
}