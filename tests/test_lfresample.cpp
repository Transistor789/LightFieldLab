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
	LFIO load;
	auto lf_img = load.read_image("../data/toy.lfr");
	auto wht_img = load.read_image("../data/MOD_0015.RAW");
	cv::Mat wht_img_gray;
	cv::demosaicing(wht_img, wht_img_gray, cv::COLOR_BayerGR2GRAY);
	wht_img_gray.convertTo(wht_img_gray, CV_8U, 255.0 / 1023.0);
	// imshowRaw("", wht_img_gray);
	// cv::waitKey();

	LFCalibrate cali(wht_img_gray);
	cali.run();

	ISPConfig config;
	config.awb_gains =
		Config::Get().img_meta()["awb"].get<std::vector<float>>();
	LFIsp<uint16_t> isp(config, lf_img, wht_img);

	Timer timer;
	// isp.lsc_awb_fused_fast().demosaic();
	isp.preview(1.5);
	timer.stop();
	timer.print_elapsed_ms();
	// cv::imshow("", isp.getResult());
	// cv::waitKey();

	auto demosaic_img = isp.getPreviewResult();
	// cv::imshow("", demosaic_img);
	// cv::waitKey();

	std::vector<cv::Mat> slice_maps; // 里面存了 x, y, x, y...
	int winSize = 0;

	// 调用读取
	LFIO::loadLookUpTables("../../data/calibration/slice_1.bin", slice_maps,
						   winSize);
	cv::Mat slice_x = slice_maps[0];
	cv::Mat slice_y = slice_maps[1];
	// cv::Mat slice_map_x, slice_map_y;
	// cali.computeSliceMaps(1, slice_map_x, slice_map_y);

	cv::Mat sliced_img;

	timer.start();
	cv::remap(demosaic_img, sliced_img, slice_x, slice_y, cv::INTER_CUBIC,
			  cv::BORDER_REPLICATE);
	timer.stop();
	timer.print_elapsed_ms();
	cv::imshow("sliced_img", sliced_img);
	// cv::waitKey();

	std::vector<cv::Mat> dehex_maps;
	LFIO::loadLookUpTables("../../data/calibration/dehex.bin", dehex_maps,
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
	LFIO load;
	auto lf_img = load.read_image("../data/toy.lfr");
	auto wht_img = load.read_image("../data/MOD_0015.RAW");
	cv::Mat wht_img_gray;
	cv::demosaicing(wht_img, wht_img_gray, cv::COLOR_BayerGR2GRAY);
	wht_img_gray.convertTo(wht_img_gray, CV_8U, 255.0 / 1023.0);

	// LFCalibrate cali(wht_img_gray);
	// cali.run();

	ISPConfig config;
	config.awb_gains =
		Config::Get().img_meta()["awb"].get<std::vector<float>>();
	LFIsp<uint16_t> isp(config, lf_img, wht_img);

	int slice_size = 0, dehex_size = 0;
	LFIO::loadLookUpTables("../../data/calibration/slice_7.bin", isp.maps.slice,
						   slice_size);
	LFIO::loadLookUpTables("../../data/calibration/dehex.bin", isp.maps.dehex,
						   dehex_size);

	Timer timer;

	isp.preview(1.0).resample();
	// isp.raw_process_fast().demosaic().resample();

	timer.stop();
	timer.print_elapsed_ms();

	const auto &sais = isp.getSAIS();
	for (int i = 0; i < sais.size(); ++i) {
		// cv::imshow("", sais[i]);
		imshowRaw("", sais[i]);
		cv::waitKey();
	}
}

int main() {
	// fast_review();
	get_sai();

	return 0;
}