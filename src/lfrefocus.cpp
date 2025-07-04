#include "lfrefocus.h"

LFRefocus::LFRefocus(QObject* parent) : QObject(parent) {}
void LFRefocus::printThreadId() {
	std::cout << "LFRefocus threadId: " << QThread::currentThreadId()
			  << std::endl;
}
void LFRefocus::init(const LightFieldPtr& ptr) {
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

void LFRefocus::onUpdateLF(const LightFieldPtr& ptr) {
	lf.reset();
	lf = ptr;
	init(ptr);
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
void LFRefocus::refocus(float alpha, int crop) {
	if (lf == nullptr) {
		qDebug() << "lf is nullptr!\n";
		return;
	}
	auto start = std::chrono::high_resolution_clock::now();

	float factor = 1.0f - 1.0f / alpha;
	int divisor = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	cv::Mat temp, sum, xq, yq;
	cv::UMat temp_gpu, sum_gpu, xq_gpu, yq_gpu;
	sum = cv::Mat(_size, _type, cv::Scalar(0));
	_refocusedImage = cv::Mat(_size, _type, cv::Scalar(0)); // 清除结果

	if (_isGpu) {
		sum_gpu = cv::UMat(_size, _type, cv::Scalar(0));
	}

	for (int i = 0; i < lf->size; i++) {
		int row = i / lf->rows;
		int col = i % lf->cols;
		if (row < crop || col < crop || row >= lf->rows - crop
			|| col >= lf->cols - crop) {
			continue;
		}
		if (_isGpu) {
			cv::add(_ymap_gpu, factor * (col - _center), yq_gpu);
			cv::add(_xmap_gpu, factor * (row - _center), xq_gpu);
			cv::remap(lf->data_gpu[i], temp_gpu, yq_gpu, xq_gpu,
					  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
			cv::add(temp_gpu, sum_gpu, sum_gpu);
		} else {
			cv::add(_ymap, factor * (col - _center), yq);
			cv::add(_xmap, factor * (row - _center), xq);
			cv::remap(lf->data[i], temp, yq, xq, cv::INTER_LINEAR,
					  cv::BORDER_REPLICATE);
			cv::add(temp, sum, sum);
		}
	}
	if (_isGpu) {
		cv::divide(sum_gpu, divisor, sum_gpu);
		_refocusedImage = sum_gpu.getMat(cv::ACCESS_READ).clone(); // clone
	} else {
		cv::divide(sum, divisor, _refocusedImage);
	}

	cv::Mat result;
	_refocusedImage.convertTo(result, CV_8UC(lf->channels));
	emit finished(result);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	qDebug() << "Refocus finished! "
			 << ", Elapsed time: " << duration.count() << " ms";
}
