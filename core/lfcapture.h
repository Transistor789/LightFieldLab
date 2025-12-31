#ifndef LFCAPTURE_H
#define LFCAPTURE_H

#include <opencv2/opencv.hpp>

class LFCapture {
public:
	explicit LFCapture();
	cv::Mat getFrame();

	static std::vector<int> getAvailableCameras(int maxSearch = 10);

	cv::VideoCapture cap;

	int start_x, start_y;
	int width, height;
};

#endif