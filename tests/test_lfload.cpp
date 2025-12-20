#include "config.h"
#include "lfio.h"
#include "utils.h"

#include <omp.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

void test_read() {
	LFIO lfpreader = LFIO();

	auto white_img = lfpreader.read_image("../data/MOD_0015.RAW");
	std::cout << white_img.at<float>(0, 0) << std::endl;

	auto lf_img = lfpreader.read_image("../data/toy.lfr");
	std::cout << lf_img.at<float>(0, 0) << std::endl;
	std::cout << Config::Get().img_meta()["camera"]["serialNumber"]
			  << std::endl;

	auto test_img = lfpreader.read_image("../data/zutomayo.jpg");

	auto lfptr = lfpreader.read_sai("../data/toy");
	cv::Mat center = lfptr->getCenter();
	std::cout << center.at<float>(200, 200) << std::endl;

	std::cout << Config::Get().app_cfg() << std::endl;
	std::cout << Config::Get().img_meta() << std::endl;
	std::cout << Config::Get().calibs() << std::endl;

	cv::Mat white_resized, lf_resized;
	cv::resize(white_img, white_resized, cv::Size(), 0.1, 0.1,
			   cv::INTER_LINEAR);
	cv::resize(lf_img, lf_resized, cv::Size(), 0.1, 0.1, cv::INTER_LINEAR);
	cv::imshow("1", white_resized);
	cv::imshow("2", lf_resized);
	cv::imshow("3", center);
	cv::imshow("4", test_img);
	cv::waitKey();
}
void test_openmp() {
	LFIO lfpreader;

	Timer timer;
	auto lf = lfpreader.read_sai("../data/toy_lftoolbox");
	timer.stop();
	timer.print_elapsed_ms();
	// cv::imshow("", lf->getCenter());
	// cv::waitKey();

	timer.start();
	lf = lfpreader.read_sai("../data/toy_lftoolbox");
	timer.stop();
	timer.print_elapsed_ms();
	// cv::imshow("", lf->getCenter());
	// cv::waitKey();
}
int main(int argc, char *argv[]) {
	// test_read();
	test_openmp();

	return 0;
}