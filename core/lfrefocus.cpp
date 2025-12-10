#include "lfrefocus.h"

void LFRefocus::init(const LfPtr &ptr) {
	if (ptr->data[0].size() == _size && ptr->data[0].type() == _type) {
		return;
	}
	_size = ptr->data[0].size();
	_type = ptr->data[0].type();
	_center = (1 + ptr->rows) / 2 - 1;

	_xmap = cv::Mat(ptr->height, 1, CV_32FC1);
	_ymap = cv::Mat(1, ptr->width, CV_32FC1);
	for (int x = 0; x < ptr->height; x++) {
		_xmap.at<float>(x, 0) = static_cast<float>(x);
	}
	for (int y = 0; y < ptr->width; y++) {
		_ymap.at<float>(0, y) = static_cast<float>(y);
	}
	_xmap = cv::repeat(_xmap, 1, ptr->width);
	_ymap = cv::repeat(_ymap, ptr->height, 1);
}

void LFRefocus::update(const LfPtr &ptr) {
	lf = ptr;
	init(lf);
}

cv::Mat LFRefocus::refocus(float alpha, int crop) {
	if (lf == nullptr) {
		std::cout << "lf is nullptr!\n";
		return cv::Mat{};
	}

	float factor = 1.0f - 1.0f / alpha;
	int divisor = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	cv::Mat xq, yq, temp, sum;
	sum = cv::Mat(_size, _type, cv::Scalar(0));
	for (int i = 0; i < lf->size; i++) {
		int row = i / lf->rows;
		int col = i % lf->cols;
		if (row < crop || col < crop || row >= lf->rows - crop
			|| col >= lf->cols - crop) {
			continue;
		}
		yq = _ymap + factor * (col - _center);
		xq = _xmap + factor * (row - _center);
		cv::remap(lf->data[i], temp, yq, xq, cv::INTER_LINEAR,
				  cv::BORDER_REPLICATE);
		sum = sum + temp;
	}
	cv::Mat result = sum / divisor;
	result.convertTo(result, CV_8UC(lf->channels), 255.0);

	return result;
}
