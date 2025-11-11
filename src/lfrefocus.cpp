#include "lfrefocus.h"

// LFRefocus::LFRefocus(QObject* parent) : QObject(parent) {}
LFRefocus::~LFRefocus() {
	_xmap_gpu.release();
	_ymap_gpu.release();
}

void LFRefocus::init(const LightFieldPtr &ptr) {
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

void LFRefocus::onUpdateLF(const LightFieldPtr &ptr) {
	lf = ptr;
	init(lf);
}
void LFRefocus::setGpu(bool isGPU) {
	_isGpu = isGPU;

	if (_isGpu) {
		_xmap_gpu = _xmap.getUMat(cv::ACCESS_READ);
		_ymap_gpu = _ymap.getUMat(cv::ACCESS_READ);
	} else {
		_xmap_gpu.release();
		_ymap_gpu.release();
	}
}
int LFRefocus::refocus(cv::Mat &img, float alpha, int crop) {
	if (lf == nullptr) {
		qDebug() << "lf is nullptr!\n";
		return -1;
	}

	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat temp;
	if (_isGpu) {
		refocus_gpu(temp, alpha, crop);
	} else {
		refocus_cpu(temp, alpha, crop);
	}

	temp.convertTo(img, CV_8UC(lf->channels), 255.0);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	if (_isGpu) {
		qDebug() << "Refocus by GPU"
				 << ", Elapsed time: " << duration.count() << " ms";
	} else {
		qDebug() << "Refocus by CPU"
				 << ", Elapsed time: " << duration.count() << " ms";
	}

	return 0;
}
int LFRefocus::refocus_cpu(cv::Mat &result, float alpha, int crop) {
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
	result = sum / divisor;
	return 0;
}
int LFRefocus::refocus_gpu(cv::Mat &result, float alpha, int crop) {
	float factor = 1.0f - 1.0f / alpha;
	int divisor = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	cv::UMat yq_gpu, xq_gpu, temp_gpu, result_gpu;
	cv::UMat sum_gpu(_size, _type, cv::Scalar(0));
	for (int i = 0; i < lf->size; i++) {
		int row = i / lf->rows;
		int col = i % lf->cols;
		if (row < crop || col < crop || row >= lf->rows - crop
			|| col >= lf->cols - crop) {
			continue;
		}

		cv::add(_ymap_gpu, factor * (col - _center), yq_gpu);
		cv::add(_xmap_gpu, factor * (row - _center), xq_gpu);
		cv::remap(lf->data_gpu[i], temp_gpu, yq_gpu, xq_gpu, cv::INTER_LINEAR,
				  cv::BORDER_REPLICATE);
		cv::add(temp_gpu, sum_gpu, sum_gpu);
	}
	cv::divide(sum_gpu, divisor, result_gpu);
	result = result_gpu.getMat(cv::ACCESS_READ).clone(); // clone
	return 0;
}
