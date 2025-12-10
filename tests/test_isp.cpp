#include "isp.hpp"
#include "lfload.h"
#include "utils.h"

#include <cstdint>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

void simple_test() {
	cv::Mat img = cv::Mat::ones(512, 512, CV_8U) * 128;
	img.at<uint8_t>(128, 128) = 255;
	img.at<uint8_t>(256, 128) = 255;
	img.at<uint8_t>(128, 256) = 255;
	img.at<uint8_t>(256, 256) = 255;
	cv::imshow("raw", img);

	ISPPipeline<uint8_t> isp(img);
	isp.blc({5, 5, 5, 5}, {255, 255, 255, 255});
	cv::imshow("blc", img);

	isp.dpc();
	cv::imshow("dpc", img);

	isp.awb();
	cv::imshow("awb", img);
	cv::waitKey();
}
int main() {
	LFLoad load;
	auto lf_img = load.read_image("../data/toy.lfr");
	writeJson("../../data/toy.json", load.json_dict);

	ISPPipeline<uint16_t> isp(lf_img);
	auto black_levels =
		Config::Get().img_meta()["blc"]["black"].get<std::vector<uint16_t>>();
	auto white_levels =
		Config::Get().img_meta()["blc"]["white"].get<std::vector<uint16_t>>();
	auto gains = Config::Get().img_meta()["awb"].get<std::vector<float>>();
	auto ccm = Config::Get().img_meta()["ccm"].get<std::vector<float>>();
	auto gamma = Config::Get().img_meta()["gam"].get<float>();

	saveAs8Bit("../../data/toy_original.png", isp.getResult());

	isp.blc(black_levels, white_levels);
	saveAs8Bit("../../data/toy_blc.png", isp.getResult());

	isp.dpc();
	saveAs8Bit("../../data/toy_dpc.png", isp.getResult());

	isp.awb(gains);
	saveAs8Bit("../../data/toy_awb.png", isp.getResult());

	isp.demosaic(true);
	saveAs8Bit("../../data/toy_demosaic.png", isp.getResult());

	isp.ccm(ccm);
	saveAs8Bit("../../data/toy_ccm.png", isp.getResult());

	isp.gamma(1.0f / gamma / 2);
	saveAs8Bit("../../data/toy_gamma.png", isp.getResult());

	return 0;
}