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
	LFIO load;
	json j;
	auto lf_img = load.readLFP("../data/toy.lfr", &j);
	auto wht_img = load.readLFP("../data/MOD_0015.RAW");
	// lf_img.convertTo(lf_img, CV_8U, 255.0 / 1023.0);
	// wht_img.convertTo(wht_img, CV_8U, 255.0 / 1023.0);

	LFIsp isp(j, lf_img, wht_img);

	// isp.preview(1.0);
	// cv::imshow("", isp.getPreviewResult());
	// cv::waitKey();

	isp.dpc_fast().blc_fast().lsc_awb_fused_fast().demosaic().ccm_fast();
	cv::Mat img;
	isp.getResult().convertTo(img, CV_8U, 255.0 / 1023.0);
	cv::imshow("", img);
	cv::waitKey();
}
int main() {
	test_isp();

	return 0;
}