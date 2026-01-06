#include "lfio.h"
#include "lfparams.h"
#include "lfsr.h"
#include "utils.h"

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

int main() {
#ifdef WIN32
	std::string path = "../data/toy_lftoolbox";
#else
	std::string path = "/Users/jax/code/LightFieldLab/input/toy";
#endif
	auto lf = LFIO::readSAI(path);
	auto center = lf->getCenter();
	cv::imshow("ORIGINAL", center);

	LFSuperRes sr;
	sr.setScale(2);

	auto res_cubic = sr.upsample(center, LFSuperRes::Method::CUBIC);
	cv::imshow("res_cubic", res_cubic);

	auto res_espcn = sr.upsample(center, LFSuperRes::Method::ESPCN);
	cv::imshow("res_espcn", res_espcn);

	sr.setPatchSize(196);
	sr.setScale(2);
	auto res_distgssrx2 = sr.upsample(lf->data, LFSuperRes::Method::DISTGSSR);
	cv::imshow("DISTGSSR x2", res_distgssrx2[res_distgssrx2.size() / 2]);

	sr.setScale(4);
	auto res_distgssrx4 = sr.upsample(lf->data, LFSuperRes::Method::DISTGSSR);
	cv::imshow("DISTGSSR x4", res_distgssrx4[res_distgssrx4.size() / 2]);

	cv::waitKey();

	return 0;
}