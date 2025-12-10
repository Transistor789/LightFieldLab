#include <chrono>
#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/opencv.hpp>

int main() {
	std::cout << cv::getVersionString() << std::endl;
	cv::dnn_superres::DnnSuperResImpl sr;

#ifdef WIN32
	cv::Mat input =
		cv::imread("../data/toy/input_Cam112.png", cv::IMREAD_COLOR);
	sr.readModel("../data/opencv_srmodel/FSRCNN_x2.pb");
#else
	cv::Mat input =
		cv::imread("/Users/jax/code/LightFieldLab/input/toy/input_Cam112.png",
				   cv::IMREAD_COLOR);
	sr.readModel(
		"/Users/jax/code/LightFieldLab/input/opencv_srmodel/FSRCNN_x2.pb");
#endif

	sr.setModel("fsrcnn", 2);

	cv::Mat output;

	auto start = std::chrono::high_resolution_clock::now();
	sr.upsample(input, output);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "Upsampling took: " << duration.count() << " ms" << std::endl;

	cv::imshow("1", input);
	cv::imshow("2", output);
	cv::waitKey(0);

	return 0;
}