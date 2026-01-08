#include "lfrefocus.h"

#include <algorithm> // 用于 std::max
#include <iostream>	 // 用于 std::cerr
#include <omp.h>
#include <thread>


void LFRefocus::init(const std::shared_ptr<LFData> &ptr) {
	if (ptr->data.empty())
		return;

	// [新增] 强制检查 8-bit
	if (ptr->data[0].depth() != CV_8U) {
		std::cerr << "[LFRefocus] Error: Only 8-bit (CV_8U) data is supported. "
					 "Initialization failed."
				  << std::endl;
		_size = cv::Size(0, 0); // 标记为未初始化
		return;
	}

	// 如果尺寸和类型没变，直接返回
	if (ptr->data[0].size() == _size && ptr->data[0].type() == _type) {
		return;
	}

	_size = ptr->data[0].size();
	_type = ptr->data[0].type(); // 这里 _type 必定是 CV_8UCn
	_center = (1 + ptr->rows) / 2 - 1;

	// 预计算坐标映射表 (CV_32F)
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

void LFRefocus::setLF(const std::shared_ptr<LFData> &ptr) {
	lf = ptr;
	init(lf);
}

cv::Mat LFRefocus::refocusByAlpha(float alpha, int crop) {
	// 复用 refocusByShift 的逻辑，转换 alpha 到 shift
	// shift = 1.0 - 1.0 / alpha
	// 但为了保持代码独立性，这里还是单独写，逻辑与 ByShift 一致
	return refocusByShift(1.0f - 1.0f / alpha, crop);
}

cv::Mat LFRefocus::refocusByShift(float shift, int crop) {
	// 【安全检查 1】指针和数据有效性
	if (lf == nullptr || lf->data.empty()) {
		std::cerr << "[LFRefocus] Error: LightField data is empty!"
				  << std::endl;
		return cv::Mat();
	}

	// 【安全检查 2】确保初始化成功且是 8-bit
	if (_xmap.empty() || _size.width == 0 || lf->data[0].depth() != CV_8U) {
		std::cerr << "[LFRefocus] Error: Not initialized or invalid bit depth "
					 "(require 8-bit)."
				  << std::endl;
		return cv::Mat();
	}

	int old_num_threads = cv::getNumThreads();
	cv::setNumThreads(0);

	unsigned int concurrent_threads = std::thread::hardware_concurrency();
	int num_strips = std::max(1u, concurrent_threads);

	float factor = shift;

	int count_views = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	if (count_views <= 0)
		count_views = 1;

	// 准备累加器 (CV_32F)
	int channels = lf->data[0].channels();
	int float_type = CV_MAKETYPE(CV_32F, channels);
	cv::Mat global_sum = cv::Mat::zeros(_size, float_type);

	int strip_h = _size.height / num_strips;
	if (strip_h == 0) {
		num_strips = 1;
		strip_h = _size.height;
	}

	std::atomic<bool> has_error(false);

#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_strips; ++s) {
		if (has_error)
			continue;

		try {
			int start_y = s * strip_h;
			int end_y =
				(s == num_strips - 1) ? _size.height : (start_y + strip_h);
			int current_strip_h = end_y - start_y;

			if (current_strip_h > 0) {
				cv::Rect roi(0, start_y, _size.width, current_strip_h);

				cv::Mat map_y_strip = _ymap(roi);
				cv::Mat map_x_strip = _xmap(roi);
				cv::Mat sum_strip = global_sum(roi);
				cv::Mat temp_strip, yq, xq;

				for (int i = 0; i < lf->size; i++) {
					if (i >= lf->data.size())
						break;

					int row = i / lf->rows;
					int col = i % lf->cols;

					if (row < crop || col < crop || row >= lf->rows - crop
						|| col >= lf->cols - crop)
						continue;

					float shift_y = factor * (col - _center);
					float shift_x = factor * (row - _center);

					cv::add(map_y_strip, cv::Scalar(shift_y), yq);
					cv::add(map_x_strip, cv::Scalar(shift_x), xq);

					// Remap: 输入 8-bit -> 输出 8-bit (temp_strip)
					// INTER_LINEAR 可能会产生浮点结果，但目标 temp_strip
					// 未指定类型时，remap 默认与 src 一致(8U)
					// 为了精度，这里其实可以让 remap 输出
					// 32F，但为了性能和兼容，通常输出 8U 后再 accumulate
					cv::remap(lf->data[i], temp_strip, yq, xq, cv::INTER_LINEAR,
							  cv::BORDER_REPLICATE);

					// 累加: 8U -> 32F
					cv::accumulate(temp_strip, sum_strip);
				}
			}
		} catch (...) {
			has_error = true;
		}
	}

	cv::setNumThreads(old_num_threads);

	if (has_error)
		return cv::Mat();

	// 结果转换：32F -> 8U
	cv::Mat result;
	// 强制转换为 8-bit 类型 (CV_8UCn)
	int output_type = CV_MAKETYPE(CV_8U, channels);
	global_sum.convertTo(result, output_type, 1.0 / count_views);

	return result;
}

cv::Mat LFRefocus::generateAllInFocus(float min_shift, float max_shift,
									  float step, int crop) {
	if (lf == nullptr)
		return cv::Mat();

	std::vector<cv::Mat> stack;
	// 简单预估大小
	int approx_count = static_cast<int>((max_shift - min_shift) / step) + 1;
	stack.reserve(approx_count > 0 ? approx_count : 1);

	for (float s = min_shift; s <= max_shift; s += step) {
		cv::Mat layer = refocusByShift(s, crop);
		if (!layer.empty()) {
			stack.push_back(layer);
		}
	}
	return mergeFocalStack(stack);
}

cv::Mat LFRefocus::mergeFocalStack(const std::vector<cv::Mat> &stack) {
	if (stack.empty())
		return cv::Mat();

	// 检查第一张图的位深
	if (stack[0].depth() != CV_8U) {
		std::cerr << "[LFRefocus] Error: mergeFocalStack requires 8-bit images."
				  << std::endl;
		return cv::Mat();
	}

	cv::Size size = stack[0].size();
	int type = stack[0].type(); // 8UCn

	cv::Mat result = cv::Mat::zeros(size, type);
	cv::Mat maxScore = cv::Mat::zeros(size, CV_64F);

	for (const auto &slice : stack) {
		if (slice.depth() != CV_8U)
			continue; // 跳过非法帧

		cv::Mat gray, score;
		if (slice.channels() == 3) {
			cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
		} else {
			gray = slice;
		}

		// 计算梯度 (Laplacian 输出 16S 或 64F，这里用 64F 保证精度)
		cv::Laplacian(gray, score, CV_64F, 3);
		score = cv::abs(score);
		cv::GaussianBlur(score, score, cv::Size(3, 3), 0);

		// Winner Takes All 融合
		for (int r = 0; r < size.height; ++r) {
			double *ptrScore = score.ptr<double>(r);
			double *ptrMaxScore = maxScore.ptr<double>(r);

			// 输入输出都是 8-bit (uchar)
			const uchar *ptrSrc = slice.ptr<uchar>(r);
			uchar *ptrDst = result.ptr<uchar>(r);

			int channels = slice.channels();

			for (int c = 0; c < size.width; ++c) {
				if (ptrScore[c] > ptrMaxScore[c]) {
					ptrMaxScore[c] = ptrScore[c];
					for (int k = 0; k < channels; ++k) {
						ptrDst[c * channels + k] = ptrSrc[c * channels + k];
					}
				}
			}
		}
	}
	return result;
}