#include "lfio.h"
#include "lfsr.h"

#include <chrono>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

int main() {
#ifdef WIN32
	std::string path = "../data/toy_lftoolbox";
#else
	std::string path = "/Users/jax/code/LightFieldLab/input/toy";
#endif
	auto lf = LFIO::read_sai(path);
	auto center = lf->getCenter();
	cv::imshow("ORIGINAL", center);

	LFSuperRes sr;
	sr.setScale(2);
	sr.setType(ModelType::CUBIC);

	auto res_cubic = sr.upsample(center);
	cv::imshow("res_cubic", res_cubic);

	sr.setType(ModelType::ESPCN);
	auto res_espcn = sr.upsample(center);
	cv::imshow("res_espcn", res_espcn);

	sr.setType(ModelType::DISTGSSR);
	sr.setLF(lf);
	sr.setPatchSize(128);
	// sr.se
	auto res_distgssr = sr.upsample();
	cv::imshow("DISTGSSR", res_distgssr);

	cv::waitKey();

	return 0;
}