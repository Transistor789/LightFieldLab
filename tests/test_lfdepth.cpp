#include "lfdepth.h"
#include "lfio.h"
#include "utils.h"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main() {
	auto lf = LFIO::readSAI("../data/bedroom");
	LFDisp disp;

	disp.setPatchSize(196);
	Timer timer;
	disp.depth(lf->data);
	timer.stop();
	timer.print_elapsed_ms();

	timer.start();
	disp.depth(lf->data);
	timer.stop();
	timer.print_elapsed_ms();

	// if (disp.hasResult()) {
	// 	auto plasma = disp.getPlasmaVisual();
	// 	cv::imshow("plasma", plasma);

	// 	auto jet = disp.getJetVisual();
	// 	cv::imshow("jet", jet);

	// 	auto gray = disp.getGrayVisual();
	// 	cv::imshow("gray", gray);

	// 	cv::waitKey();
	// }

	disp.setPatchSize(128);
	timer.start();
	disp.depth(lf->data);
	timer.stop();
	timer.print_elapsed_ms();

	timer.start();
	disp.depth(lf->data);
	timer.stop();
	timer.print_elapsed_ms();

	// if (disp.hasResult()) {
	// 	auto plasma = disp.getPlasmaVisual();
	// 	cv::imshow("plasma", plasma);

	// 	auto jet = disp.getJetVisual();
	// 	cv::imshow("jet", jet);

	// 	auto gray = disp.getGrayVisual();
	// 	cv::imshow("gray", gray);

	// 	cv::waitKey();
	// }

	return 0;
}