#include "lfio.h"
#include "lfrefocus.h"
#include "utils.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
	auto lf = LFIO::ReadSAI("../data/toy_lftoolbox");

	LFRefocus rfc;
	rfc.setLF(lf);

	Timer timer;
	auto result = rfc.refocusByAlpha(1);
	timer.stop();
	timer.print_elapsed_ms();
	cv::imshow("result", result);
	cv::waitKey();

	return 0;
}