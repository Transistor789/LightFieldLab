#include "config.h"
#include "lfio.h"
#include "lfisp.h"
#include "utils.h"

#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

void test_isp() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadLFP("../data/MOD_0015.RAW", meta);
	// lf_img.convertTo(lf_img, CV_8U, 255.0 / 1023.0);
	// wht_img.convertTo(wht_img, CV_8U, 255.0 / 1023.0);

	IspConfig config;
	LFIsp::parseJsonToConfig(j, config);
	LFIsp isp(lf_img, wht_img, config);

	// isp.preview(1.0);
	// cv::imshow("", isp.getPreviewResult());
	// cv::waitKey();

	isp.dpc_fast(DpcMethod::Diretional, 100)
		.blc_fast(66)
		.lsc_awb_fused_fast(0, {})
		.demosaic(config.bayer, DemosaicMethod::Bilinear)
		.ccm_fast({});
	cv::Mat img;
	isp.getResult().convertTo(img, CV_8U, 255.0 / 1023.0);
	cv::imshow("", img);
	cv::waitKey();
}

void test_speed() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadWhiteImageManual("../data/MOD_0015.RAW", meta);

	LFIsp isp;
	IspConfig config;
	isp.parseJsonToConfig(j, config);
	isp.set_lf_img(lf_img).initConfig(wht_img, config);

	Timer timer;
	isp.preview(config);
	// isp.awb_fast({1.0, 1.0, 1.0, 1.0});
	timer.stop();
	timer.print_elapsed_ms();

	// cv::imshow("", isp.getResult());
	// cv::waitKey();
}

int main() {
	// test_isp();
	test_speed();

	return 0;
}