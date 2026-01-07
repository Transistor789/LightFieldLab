#include "lfrefocus.h"

#include <algorithm> // 用于 std::max
#include <omp.h>
#include <thread>

void LFRefocus::init(const std::shared_ptr<LFData> &ptr) {
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

void LFRefocus::setLF(const std::shared_ptr<LFData> &ptr) {
	lf = ptr;
	init(lf);
}

cv::Mat LFRefocus::refocusByAlpha(float alpha, int crop) {
	// 【安全检查 1】指针和数据有效性
	if (lf == nullptr || lf->data.empty()) {
		std::cerr << "[Error] LightField data is empty!" << std::endl;
		return cv::Mat();
	}

	// 【安全检查 2】确保 Map 已初始化
	if (_xmap.empty() || _ymap.empty() || _xmap.size() != _size) {
		std::cerr
			<< "[Error] Remap coordinate maps not initialized or size mismatch!"
			<< std::endl;
		return cv::Mat();
	}

	// 保存并关闭 OpenCV 内部多线程，防止与 OpenMP 冲突
	int old_num_threads = cv::getNumThreads();
	cv::setNumThreads(0);

	unsigned int concurrent_threads = std::thread::hardware_concurrency();
	int num_strips = std::max(1u, concurrent_threads);

	float factor = 1.0f - 1.0f / alpha;

	// 计算分母，防止除以0
	int count_views = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	if (count_views <= 0)
		count_views = 1;

	// 【核心修复 1】累加器必须使用浮点型 (CV_32F)，否则 8位 累加瞬间溢出或报错
	// 假设输入是 3通道，则 _type 可能是 CV_8UC3，这里强制转为 CV_32FC3
	int channels = CV_MAT_CN(_type);
	int float_type = CV_MAKETYPE(CV_32F, channels);
	cv::Mat global_sum = cv::Mat::zeros(_size, float_type);

	int strip_h = _size.height / num_strips;
	if (strip_h == 0) {
		num_strips = 1;
		strip_h = _size.height;
	}

	// 标记是否有错误发生
	std::atomic<bool> has_error(false);

#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_strips; ++s) {
		// 如果其他线程崩了，这里尽快退出
		if (has_error)
			continue;

		try {
			int start_y = s * strip_h;
			int end_y =
				(s == num_strips - 1) ? _size.height : (start_y + strip_h);
			int current_strip_h = end_y - start_y;

			if (current_strip_h > 0) {
				// 构造 ROI
				cv::Rect roi(0, start_y, _size.width, current_strip_h);

				// 获取引用 (注意：read-only map 不需要深拷贝)
				cv::Mat map_y_strip = _ymap(roi);
				cv::Mat map_x_strip = _xmap(roi);
				cv::Mat sum_strip = global_sum(roi);

				// 线程局部变量
				cv::Mat temp_strip, yq, xq;

				for (int i = 0; i < lf->size; i++) {
					// 防止越界
					if (i >= lf->data.size())
						break;

					int row = i / lf->rows;
					int col = i % lf->cols;

					// 裁剪逻辑
					if (row < crop || col < crop || row >= lf->rows - crop
						|| col >= lf->cols - crop)
						continue;

					float shift_y = factor * (col - _center);
					float shift_x = factor * (row - _center);

					// 计算偏移后的坐标 (这里会分配内存，是性能热点，但难以避免)
					// 注意：add 运算会自动处理 broadcasting
					cv::add(map_y_strip, cv::Scalar(shift_y), yq);
					cv::add(map_x_strip, cv::Scalar(shift_x), xq);

					// Remap
					// 注意：remap 的输出 temp_strip 类型通常跟随 input 类型
					// (CV_8U)
					cv::remap(lf->data[i], temp_strip, yq, xq, cv::INTER_LINEAR,
							  cv::BORDER_REPLICATE);

					// 累加：accumulate 会自动处理 8U -> 32F 的转换
					cv::accumulate(temp_strip, sum_strip);
				}
			}
		} catch (const cv::Exception &e) {
			// 【核心修复 2】捕获 OpenCV 异常，防止 OpenMP 直接崩溃
			std::cerr << "[OpenMP Error] OpenCV Exception in thread " << s
					  << ": " << e.what() << std::endl;
			has_error = true;
		} catch (const std::exception &e) {
			std::cerr << "[OpenMP Error] Std Exception in thread " << s << ": "
					  << e.what() << std::endl;
			has_error = true;
		} catch (...) {
			std::cerr << "[OpenMP Error] Unknown Exception in thread " << s
					  << std::endl;
			has_error = true;
		}
	}

	// 恢复线程设置
	cv::setNumThreads(old_num_threads);

	if (has_error) {
		return cv::Mat(); // 或者返回空图表示失败
	}

	// 结果转换：32F -> 8U
	cv::Mat result;
	// scale = 1.0 / count_views, 这里的 convertTo 会自动做 saturate_cast
	// 不需要手动乘以 255，因为累加的是原始像素值 (0-255)，除以数量后还是
	// (0-255) 如果你之前的逻辑是 归一化到0-1了，请检查这里 假设输入是 0-255 的
	// 8U，则直接除以数量即可
	global_sum.convertTo(result, CV_8UC(channels), 1.0 / count_views);

	return result;
}
cv::Mat LFRefocus::refocusByShift(float shift, int crop) {
	// 【安全检查 1】指针和数据有效性
	if (lf == nullptr || lf->data.empty()) {
		std::cerr << "[Error] LightField data is empty!" << std::endl;
		return cv::Mat();
	}

	// 【安全检查 2】确保 Map 已初始化
	if (_xmap.empty() || _ymap.empty() || _xmap.size() != _size) {
		std::cerr
			<< "[Error] Remap coordinate maps not initialized or size mismatch!"
			<< std::endl;
		return cv::Mat();
	}

	// 保存并关闭 OpenCV 内部多线程，防止与 OpenMP 冲突
	int old_num_threads = cv::getNumThreads();
	cv::setNumThreads(0);

	unsigned int concurrent_threads = std::thread::hardware_concurrency();
	int num_strips = std::max(1u, concurrent_threads);

	float factor = shift;

	// 计算分母，防止除以0
	int count_views = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	if (count_views <= 0)
		count_views = 1;

	// 【核心修复 1】累加器必须使用浮点型 (CV_32F)，否则 8位 累加瞬间溢出或报错
	// 假设输入是 3通道，则 _type 可能是 CV_8UC3，这里强制转为 CV_32FC3
	int channels = CV_MAT_CN(_type);
	int float_type = CV_MAKETYPE(CV_32F, channels);
	cv::Mat global_sum = cv::Mat::zeros(_size, float_type);

	int strip_h = _size.height / num_strips;
	if (strip_h == 0) {
		num_strips = 1;
		strip_h = _size.height;
	}

	// 标记是否有错误发生
	std::atomic<bool> has_error(false);

#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_strips; ++s) {
		// 如果其他线程崩了，这里尽快退出
		if (has_error)
			continue;

		try {
			int start_y = s * strip_h;
			int end_y =
				(s == num_strips - 1) ? _size.height : (start_y + strip_h);
			int current_strip_h = end_y - start_y;

			if (current_strip_h > 0) {
				// 构造 ROI
				cv::Rect roi(0, start_y, _size.width, current_strip_h);

				// 获取引用 (注意：read-only map 不需要深拷贝)
				cv::Mat map_y_strip = _ymap(roi);
				cv::Mat map_x_strip = _xmap(roi);
				cv::Mat sum_strip = global_sum(roi);

				// 线程局部变量
				cv::Mat temp_strip, yq, xq;

				for (int i = 0; i < lf->size; i++) {
					// 防止越界
					if (i >= lf->data.size())
						break;

					int row = i / lf->rows;
					int col = i % lf->cols;

					// 裁剪逻辑
					if (row < crop || col < crop || row >= lf->rows - crop
						|| col >= lf->cols - crop)
						continue;

					float shift_y = factor * (col - _center);
					float shift_x = factor * (row - _center);

					// 计算偏移后的坐标 (这里会分配内存，是性能热点，但难以避免)
					// 注意：add 运算会自动处理 broadcasting
					cv::add(map_y_strip, cv::Scalar(shift_y), yq);
					cv::add(map_x_strip, cv::Scalar(shift_x), xq);

					// Remap
					// 注意：remap 的输出 temp_strip 类型通常跟随 input 类型
					// (CV_8U)
					cv::remap(lf->data[i], temp_strip, yq, xq, cv::INTER_LINEAR,
							  cv::BORDER_REPLICATE);

					// 累加：accumulate 会自动处理 8U -> 32F 的转换
					cv::accumulate(temp_strip, sum_strip);
				}
			}
		} catch (const cv::Exception &e) {
			// 【核心修复 2】捕获 OpenCV 异常，防止 OpenMP 直接崩溃
			std::cerr << "[OpenMP Error] OpenCV Exception in thread " << s
					  << ": " << e.what() << std::endl;
			has_error = true;
		} catch (const std::exception &e) {
			std::cerr << "[OpenMP Error] Std Exception in thread " << s << ": "
					  << e.what() << std::endl;
			has_error = true;
		} catch (...) {
			std::cerr << "[OpenMP Error] Unknown Exception in thread " << s
					  << std::endl;
			has_error = true;
		}
	}

	// 恢复线程设置
	cv::setNumThreads(old_num_threads);

	if (has_error) {
		return cv::Mat(); // 或者返回空图表示失败
	}

	// 结果转换：32F -> 8U
	cv::Mat result;
	// scale = 1.0 / count_views, 这里的 convertTo 会自动做 saturate_cast
	// 不需要手动乘以 255，因为累加的是原始像素值 (0-255)，除以数量后还是
	// (0-255) 如果你之前的逻辑是 归一化到0-1了，请检查这里 假设输入是 0-255 的
	// 8U，则直接除以数量即可
	// global_sum.convertTo(result, CV_8UC(channels), 1.0 / count_views);
	global_sum.convertTo(result, _type, 1.0 / count_views);

	return result;
}

cv::Mat LFRefocus::generateAllInFocus(float min_shift, float max_shift,
									  float step, int crop) {
	if (lf == nullptr)
		return cv::Mat();

	// 1. [内部步骤 A] 生成堆栈
	// 不需要把这个 vector 暴露给外部，它是临时中间变量
	std::vector<cv::Mat> stack;
	int approx_count = static_cast<int>((max_shift - min_shift) / step) + 1;
	stack.reserve(approx_count);

	// 循环生成每一层
	for (float s = min_shift; s <= max_shift; s += step) {
		cv::Mat layer = refocusByShift(s, crop);
		if (!layer.empty()) {
			stack.push_back(layer);
		}
	}

	if (stack.empty())
		return cv::Mat();

	// 2. [内部步骤 B] 融合堆栈
	// 调用私有融合函数
	return mergeFocalStack(stack);
}

cv::Mat LFRefocus::mergeFocalStack(const std::vector<cv::Mat> &stack) {
	if (stack.empty())
		return cv::Mat();

	cv::Size size = stack[0].size();
	int type = stack[0].type();

	// 准备结果矩阵
	cv::Mat result = cv::Mat::zeros(size, type);

	// 记录每个像素位置目前为止见过的“最大清晰度分值”
	// 使用双精度浮点数保证精度
	cv::Mat maxScore = cv::Mat::zeros(size, CV_64F);

	// 遍历每一张切片
	for (const auto &slice : stack) {
		// 1. 计算当前切片的清晰度图 (Score Map)
		cv::Mat gray, score;
		if (slice.channels() == 3) {
			cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
		} else {
			gray = slice;
		}

		// 使用 Laplacian 算子提取梯度作为清晰度指标
		cv::Laplacian(gray, score, CV_64F, 3);
		score = cv::abs(score);

		// 稍微模糊一下分数图，减少噪点带来的“斑点”效应
		cv::GaussianBlur(score, score, cv::Size(3, 3), 0);

		// 2. 逐像素比较并融合 (Winner Takes All)
		// 使用指针遍历提高效率
		for (int r = 0; r < size.height; ++r) {
			double *ptrScore = score.ptr<double>(r);
			double *ptrMaxScore = maxScore.ptr<double>(r);

			// 假设是 3 通道 (CV_8UC3)，如果是单通道需要根据 type 调整
			// 这里为了通用性，可以用 uchar* + step
			const uchar *ptrSrc = slice.ptr<uchar>(r);
			uchar *ptrDst = result.ptr<uchar>(r);

			int channels = slice.channels();

			for (int c = 0; c < size.width; ++c) {
				if (ptrScore[c] > ptrMaxScore[c]) {
					// 发现更清晰的像素，更新最大分值
					ptrMaxScore[c] = ptrScore[c];

					// 拷贝颜色数据
					for (int k = 0; k < channels; ++k) {
						ptrDst[c * channels + k] = ptrSrc[c * channels + k];
					}
				}
			}
		}
	}

	return result;
}