#include "lfload.h"
#include "lfrefocus.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
	LFLoad lfreader;
	auto lf = lfreader.read_sai("../data/toy", true);

	LFRefocus rfc;
	rfc.update(lf);
	auto result = rfc.refocus(0.5);
	cv::imshow("", result);
	cv::waitKey();

	return 0;
}