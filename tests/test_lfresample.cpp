#include "config.h"
#include "lfcalibrate.h"
#include "lfio.h"
#include "lfisp.h"
#include "utils.h"

#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

void fast_preview() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadWhiteImageManual("../data/MOD_0015.RAW", meta);
	cv::Mat wht_img_gray;
	cv::demosaicing(wht_img, wht_img_gray, cv::COLOR_BayerGR2GRAY);
	wht_img_gray.convertTo(wht_img_gray, CV_8U, 255.0 / 1023.0);
	// imshowRaw("", wht_img_gray);
	// cv::waitKey();

	LFCalibrate cali(wht_img_gray);
	CalibrateConfig config;
	config.ceMethod = ExtractMethod::Contour;
	config.bayer = BayerPattern::NONE;
	config.bitDepth = 8;
	cali.run(config);

	IspConfig ispConfig;
	LFIsp::parseJsonToConfig(j, ispConfig);
	LFIsp isp(lf_img, wht_img, ispConfig);

	Timer timer;
	// isp.lsc_awb_fused_fast().demosaic();
	isp.preview(ispConfig);
	timer.stop();
	timer.print_elapsed_ms();
	// cv::imshow("", isp.getResult());
	// cv::waitKey();

	auto demosaic_img = isp.getResult();
	// cv::imshow("", demosaic_img);
	// cv::waitKey();

	std::vector<cv::Mat> slice_maps; // 里面存了 x, y, x, y...
	int winSize = 0;

	// 调用读取
	LFIO::LoadLookUpTables("../../data/calibration/slice_1.bin", slice_maps,
						   winSize);
	cv::Mat slice_x = slice_maps[0];
	cv::Mat slice_y = slice_maps[1];
	// cv::Mat slice_map_x, slice_map_y;
	// cali.computeExtractMaps(1, slice_map_x, slice_map_y);

	cv::Mat sliced_img;

	timer.start();
	cv::remap(demosaic_img, sliced_img, slice_x, slice_y, cv::INTER_CUBIC,
			  cv::BORDER_REPLICATE);
	timer.stop();
	timer.print_elapsed_ms();
	cv::imshow("sliced_img", sliced_img);
	// cv::waitKey();

	std::vector<cv::Mat> dehex_maps;
	LFIO::LoadLookUpTables("../../data/calibration/lut_dehex.bin", dehex_maps,
						   winSize);
	cv::Mat dehex_x = dehex_maps[0];
	cv::Mat dehex_y = dehex_maps[1];
	// cv::Mat dehex_x, dehex_y;
	// cali.computeDehexMaps(dehex_x, dehex_y);
	// cali.getDehexMaps(dehex_x, dehex_y);
	cv::Mat dehex_img;
	timer.start();
	cv::remap(sliced_img, dehex_img, dehex_x, dehex_y, cv::INTER_CUBIC,
			  cv::BORDER_REPLICATE);
	timer.stop();
	timer.print_elapsed_ms();
	cv::imshow("dehex_img", dehex_img);
	cv::waitKey();

	cv::Mat fused_x, fused_y;
	cv::remap(slice_x, fused_x, dehex_x, dehex_y, cv::INTER_LINEAR,
			  cv::BORDER_REPLICATE);
	cv::remap(slice_y, fused_y, dehex_x, dehex_y, cv::INTER_LINEAR,
			  cv::BORDER_REPLICATE);
	cv::Mat result;
	cv::remap(demosaic_img, result, fused_x, fused_y, cv::INTER_LINEAR,
			  cv::BORDER_REPLICATE);
	cv::imshow("fused", result);
	cv::waitKey();
}

void get_sai() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadWhiteImageManual("../data/MOD_0015.RAW", meta);
	cv::Mat wht_img_gray;
	cv::demosaicing(wht_img, wht_img_gray, cv::COLOR_BayerGR2GRAY);
	// wht_img_gray.convertTo(wht_img_gray, CV_8U, 255.0 / 1023.0);
	IspConfig config;
	LFIsp::parseJsonToConfig(j, config);
	LFIsp isp(lf_img, wht_img_gray, config);

	int slice_size = 0, dehex_size = 0;
	LFIO::LoadLookUpTables("../../data/calibration/lut_extract_9.bin",
						   isp.maps.extract, slice_size);
	LFIO::LoadLookUpTables("../../data/calibration/lut_dehex.bin",
						   isp.maps.dehex, dehex_size);

	IspConfig ispConfig;
	isp.parseJsonToConfig(j, ispConfig);

	Timer timer;
	isp.preview(ispConfig).resample(true);
	// std::cout << "test...\n";
	// isp.ccm_fast_sai();

	timer.stop();
	timer.print_elapsed_ms();

	// const auto &sais = isp.getSAIs();
	// std::cout << sais[0].size << std::endl;
	// for (int i = 0; i < sais.size(); ++i) {
	// 	// cv::Mat temp;
	// 	// sais[i].convertTo(temp, CV_8U, 255.0f / (1023 - 64));
	// 	// cv::imshow("", temp);
	// 	// cv::imwrite("../../data/center.png", temp);
	// 	cv::imshow("", sais[i]);
	// 	// cv::imwrite("../../data/center_preview.png", sais[i]);
	// 	cv::waitKey();
	// }
}

int main() {
	// fast_review();
	get_sai();

	return 0;
}