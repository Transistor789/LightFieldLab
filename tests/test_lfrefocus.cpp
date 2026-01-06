#include "lfio.h"
#include "lfrefocus.h"
#include "utils.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
	LFIO lfreader;
	auto lf = lfreader.readSAI("../data/toy_lftoolbox");

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