#ifndef ISP_H
#define ISP_H

#include <algorithm>
#include <limits> // 必须包含，用于 numeric_limits
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

// 如果您的环境编译了 OpenCV CUDA 模块，请确保包含此头文件
#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <opencv2/cudaimgproc.hpp>
#endif

enum class BayerPattern {
	GRBG, // Green-Red / Blue-Green
	RGGB, // Red-Green / Green-Blue
	GBRG, // Green-Blue / Red-Green
	BGGR  // Blue-Green / Green-Red
};

// 模板类，T 代表图像像素深度类型 (如 uint16_t, uint8_t)
template <typename T>
class ISPPipeline {
private:
	cv::Mat img_; // 内部维护的图像，所有操作都在此原地进行
	BayerPattern pattern_;
	std::vector<cv::Point> offsets_; // 预先计算好的通道偏移

	// 内部辅助函数：根据 Bayer 模式计算偏移
	std::vector<cv::Point> calculate_channel_offsets(BayerPattern pattern) {
		std::vector<cv::Point> offs(4);
		switch (pattern) {
			case BayerPattern::GRBG:
				offs[0] = cv::Point(0, 0); // Gr
				offs[1] = cv::Point(1, 0); // R
				offs[2] = cv::Point(0, 1); // B
				offs[3] = cv::Point(1, 1); // Gb
				break;
			case BayerPattern::RGGB:
				offs[0] = cv::Point(0, 0); // R
				offs[1] = cv::Point(1, 0); // Gr
				offs[2] = cv::Point(0, 1); // Gb
				offs[3] = cv::Point(1, 1); // B
				break;
			case BayerPattern::GBRG:
				offs[0] = cv::Point(0, 0); // Gb
				offs[1] = cv::Point(1, 0); // B
				offs[2] = cv::Point(0, 1); // R
				offs[3] = cv::Point(1, 1); // Gr
				break;
			case BayerPattern::BGGR:
				offs[0] = cv::Point(0, 0); // B
				offs[1] = cv::Point(1, 0); // Gb
				offs[2] = cv::Point(0, 1); // Gr
				offs[3] = cv::Point(1, 1); // R
				break;
			default:
				throw std::invalid_argument("Unsupported Bayer pattern.");
		}
		return offs;
	}

	// 辅助函数：将 BayerPattern 映射到 OpenCV 的转换代码 (Bayer -> BGR)
	int get_opencv_demosaic_code(BayerPattern pattern) const {
		switch (pattern) {
			case BayerPattern::GRBG:
				return cv::COLOR_BayerGR2BGR;
			case BayerPattern::RGGB:
				return cv::COLOR_BayerRG2BGR;
			case BayerPattern::GBRG:
				return cv::COLOR_BayerGB2BGR;
			case BayerPattern::BGGR:
				return cv::COLOR_BayerBG2BGR;
			default:
				throw std::invalid_argument(
					"Unknown Bayer Pattern for Demosaic.");
		}
	}

public:
	// 构造函数
	explicit ISPPipeline(cv::Mat img, BayerPattern pattern = BayerPattern::GRBG)
		: img_(img), pattern_(pattern) {
		// 类型检查
		if (img_.empty() || img_.channels() != 1
			|| img_.depth() != cv::DataType<T>::depth) {
			throw std::runtime_error(
				"ISPPipeline initialized with invalid image depth or channels "
				"matching template type.");
		}
		offsets_ = calculate_channel_offsets(pattern_);
	}

	// 获取处理后的图像
	cv::Mat &getResult() { return img_; }
	const cv::Mat &getResult() const { return img_; }

	// --- ISP 模块函数 ---

	// 1. Black Level Correction (BLC)
	ISPPipeline &blc(const std::vector<T> &black_levels = {64, 64, 64, 64},
					 const std::vector<T> &white_levels = {1023, 1023, 1023,
														   1023}) {
		if (black_levels.size() < 4 || white_levels.size() < 4) {
			throw std::invalid_argument(
				"Black/White levels must contain at least 4 elements.");
		}

		for (size_t i = 0; i < offsets_.size(); ++i) {
			const auto &offset = offsets_[i];
			int c_start = offset.x;
			int r_start = offset.y;
			const T BL = black_levels[i];

			for (int r = r_start; r < img_.rows; r += 2) {
				T *row_ptr = img_.template ptr<T>(r);
				for (int c = c_start; c < img_.cols; c += 2) {
					T &P = row_ptr[c];
					if (P > BL) {
						P = P - BL;
					} else {
						P = 0;
					}
				}
			}
		}
		return *this;
	}

	// 2. Defective Pixel Correction (DPC)
	ISPPipeline &dpc(int ksize = 3, float hot_threshold_factor = 0.15f,
					 float dead_threshold_factor = 0.5f) {
		if (ksize % 2 == 0 || ksize < 3) {
			throw std::invalid_argument(
				"Kernel size (ksize) must be an odd number >= 3.");
		}

		for (const auto &offset : offsets_) {
			int c_start = offset.x;
			int r_start = offset.y;

			int channel_rows = (img_.rows - r_start + 1) / 2;
			int channel_cols = (img_.cols - c_start + 1) / 2;
			cv::Mat extracted_channel(channel_rows, channel_cols, img_.type());

			for (int r = 0; r < channel_rows; ++r) {
				T *src_row_ptr = img_.template ptr<T>(r_start + r * 2);
				T *dst_row_ptr = extracted_channel.template ptr<T>(r);
				for (int c = 0; c < channel_cols; ++c) {
					dst_row_ptr[c] = src_row_ptr[c_start + c * 2];
				}
			}

			cv::Mat channel_median;
			cv::medianBlur(extracted_channel, channel_median, ksize);

			for (int r = 0; r < channel_rows; ++r) {
				T *original_row_ptr = img_.template ptr<T>(r_start + r * 2);
				T *median_row_ptr = channel_median.template ptr<T>(r);

				for (int c = 0; c < channel_cols; ++c) {
					T &P = original_row_ptr[c_start + c * 2];
					T M = median_row_ptr[c];
					float P_f = static_cast<float>(P);
					float M_f = static_cast<float>(M);

					if (P_f > M_f
						&& (P_f - M_f) > (M_f * hot_threshold_factor)) {
						P = M;
					} else if (P_f < M_f
							   && (M_f - P_f) > (M_f * dead_threshold_factor)) {
						P = M;
					}
				}
			}
		}
		return *this;
	}

	// 3. Auto White Balance (AWB)
	ISPPipeline &awb(const std::vector<float> &gains = {1.0f, 1.0f, 1.0f,
														1.0f}) {
		if (gains.size() < 4) {
			throw std::invalid_argument(
				"Gains must contain at least 4 elements.");
		}
		if (std::all_of(gains.begin(), gains.end(),
						[](float g) { return g == 1.0f; })) {
			return *this;
		}

		for (size_t i = 0; i < offsets_.size(); ++i) {
			const auto &offset = offsets_[i];
			int c_start = offset.x;
			int r_start = offset.y;
			const float gain = gains[i];

			if (gain == 1.0f)
				continue;

			for (int r = r_start; r < img_.rows; r += 2) {
				T *row_ptr = img_.template ptr<T>(r);
				for (int c = c_start; c < img_.cols; c += 2) {
					T &P = row_ptr[c];
					float P_f = static_cast<float>(P) * gain;
					P = static_cast<T>(std::min(
						P_f,
						static_cast<float>(std::numeric_limits<T>::max())));
				}
			}
		}
		return *this;
	}

	// 4. Demosaic (Bayer -> BGR)
	ISPPipeline &demosaic(bool use_cuda = false) {
		if (img_.empty() || img_.channels() != 1) {
			throw std::runtime_error(
				"Demosaic requires a 1-channel Bayer image.");
		}

		int code = get_opencv_demosaic_code(pattern_);

		if (use_cuda) {
#if defined(HAVE_OPENCV_CUDAIMGPROC) || defined(CV_CUDA)
			try {
				cv::cuda::GpuMat d_src(img_);
				cv::cuda::GpuMat d_dst;
				cv::cuda::demosaicing(d_src, d_dst, code);
				d_dst.download(img_);
			} catch (const cv::Exception &e) {
				throw std::runtime_error(std::string("CUDA Demosaic failed: ")
										 + e.what());
			}
#else
			throw std::runtime_error("OpenCV was not built with CUDA support.");
#endif
		} else {
			cv::cvtColor(img_, img_, code);
		}
		return *this;
	}

	// 5. Color Correction Matrix (CCM)
	// 输入：3通道 BGR (必须在 demosaic 之后调用)
	// ccm_matrix: 9个元素的vector (3x3 矩阵)
	ISPPipeline &ccm(const std::vector<float> &ccm_matrix) {
		if (img_.empty() || img_.channels() != 3) {
			throw std::runtime_error(
				"CCM requires a 3-channel image (run demosaic first).");
		}
		if (ccm_matrix.size() != 9) {
			throw std::invalid_argument("CCM matrix must have 9 elements.");
		}

		// 构建 3x3 矩阵
		cv::Mat m(3, 3, CV_32F);
		// 使用 memcpy 快速复制数据 (假设 vector 内存连续)
		std::memcpy(m.data, ccm_matrix.data(), 9 * sizeof(float));

		// OpenCV 的 demosaic 输出通常是 BGR
		// 大多数 CCM 是针对 RGB 定义的，因此通常做法是：BGR -> RGB -> Transform
		// -> BGR
		cv::cvtColor(img_, img_, cv::COLOR_BGR2RGB);
		cv::transform(img_, img_, m);
		cv::cvtColor(img_, img_, cv::COLOR_RGB2BGR);

		return *this;
	}

	// 6. Gamma Correction
	// 输入：gamma 值 (例如 2.2)
	// 执行公式: Output = Input ^ (1/gamma)
	ISPPipeline &gamma(float gamma_val = 2.2f, float white_level = 65535.0f) {
		if (img_.empty())
			return *this;
		if (gamma_val <= 0.0f)
			throw std::invalid_argument("Gamma value must be positive.");

		// 1. 转为浮点数并归一化到 [0, 1.0]
		// 【关键修复】使用真实的 white_level (1023) 进行归一化
		cv::Mat float_img;
		img_.convertTo(float_img, CV_32F, 1.0 / white_level);

		// 2. 执行 Gamma 幂运算: dst = src ^ (1/gamma)
		cv::pow(float_img, 1.0f / gamma_val, float_img);

		// 3. 还原回类型 T 的范围 (拉伸到满量程)
		// 我们通常希望输出图像利用满量程 (例如 10-bit 输入 -> 16-bit
		// 满量程输出) 这样显示器显示才正常
		double output_max = static_cast<double>(std::numeric_limits<T>::max());

		// 自动进行饱和度截断 (Saturate Cast)
		float_img.convertTo(img_, img_.type(), white_level);

		return *this;
	}
};

#endif // ISP_H