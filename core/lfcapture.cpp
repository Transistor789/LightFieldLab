#include "lfcapture.h"

#include "USBConfiguration.h"

#include <opencv2/core/types.hpp>

LFCapture::LFCapture() {
	int ret = SetUSBConfiguration(1920, 1080, 3, 0, 0, 0, false, false);
	if (ret == 0)
		std::cout << "USB configuration succeeded." << std::endl;
	else
		std::cout << "USB configuration failed! Error code: " << ret
				  << std::endl;

	cap = cv::VideoCapture(0, cv::CAP_DSHOW);

	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

	int cam_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

	width = 1024, height = 768;
	start_x = (cam_width - width) / 2;
	start_y = 0;
}
cv::Mat LFCapture::getFrame() {
	cv::Mat frame, gray_frame;
	cap >> frame;
	cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
	return gray_frame(cv::Rect(start_x, start_y, width, height));
}