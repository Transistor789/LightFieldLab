#include "lfio.h"
#include "lfrefocus.h"
#include "utils.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
	LFIO lfreader;
	auto lf = lfreader.read_sai("../data/toy_lftoolbox", true);

	LFRefocus rfc;
	rfc.update(lf);

	Timer timer;
	auto result = rfc.refocus(1);
	timer.stop();
	timer.print_elapsed_ms();
	cv::imshow("result", result);
	cv::waitKey();

	return 0;
}