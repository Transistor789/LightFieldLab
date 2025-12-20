#include "lfrefocus.h"

#include <algorithm> // 用于 std::max
#include <omp.h>

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

void LFRefocus::setLF(const LfPtr &ptr) {
	lf = ptr;
	init(lf);
}

cv::Mat LFRefocus::refocus(float alpha, int crop) {
	if (lf == nullptr)
		return cv::Mat();

	int old_num_threads = cv::getNumThreads();
	cv::setNumThreads(0);

	// ==========================================
	// 1. 修复核心数检测问题
	// ==========================================
	// C++标准规定 hardware_concurrency 可能返回 0
	// 我们必须给一个保底值 (比如 1)
	unsigned int concurrent_threads = std::thread::hardware_concurrency();
	int num_strips = std::max(1u, concurrent_threads);

	// 调试打印：确认这一步是否正确
	// std::cout << "[Debug] Splitting into " << num_strips << " strips." <<
	// std::endl;

	float factor = 1.0f - 1.0f / alpha;
	int divisor = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);

	// 如果没有 divisor (裁剪全裁掉了)，防止除以0
	if (divisor <= 0)
		divisor = 1;

	cv::Mat global_sum = cv::Mat::zeros(_size, _type);
	int strip_h = _size.height / num_strips;

	// 如果图像太小导致 strip_h 为 0，就只分 1 个 strip
	if (strip_h == 0) {
		num_strips = 1;
		strip_h = _size.height;
	}

#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_strips; ++s) {
		int start_y = s * strip_h;
		int end_y = (s == num_strips - 1) ? _size.height : (start_y + strip_h);
		int current_strip_h = end_y - start_y;

		if (current_strip_h <= 0)
			continue;

		cv::Rect roi(0, start_y, _size.width, current_strip_h);

		// 获取引用
		cv::Mat map_y_strip = _ymap(roi);
		cv::Mat map_x_strip = _xmap(roi);
		cv::Mat sum_strip = global_sum(roi);

		cv::Mat temp_strip, yq, xq;

		for (int i = 0; i < lf->size; i++) {
			// ... (原本的循环内容保持不变) ...
			int row = i / lf->rows;
			int col = i % lf->cols;
			if (row < crop || col < crop || row >= lf->rows - crop
				|| col >= lf->cols - crop)
				continue;

			float shift_y = factor * (col - _center);
			float shift_x = factor * (row - _center);

			// 构造 Map
			yq = map_y_strip + shift_y;
			xq = map_x_strip + shift_x;

			cv::remap(lf->data[i], temp_strip, yq, xq, cv::INTER_LINEAR,
					  cv::BORDER_REPLICATE);
			cv::accumulate(temp_strip, sum_strip);
		}
	}

	cv::setNumThreads(old_num_threads);

	cv::Mat result;
	// 确保 divisor 正确
	global_sum.convertTo(result, CV_8UC(lf->channels), 255.0 / divisor);
	return result;
}