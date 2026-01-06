#include "lfisp.h"

#include <cmath>

// ============================================================================
// 本地 Helper 模板函数 (用于标量处理的统一实现，避免代码重复)
// ============================================================================
namespace {

template <typename T>
void blc_scalar_impl(cv::Mat &img, int black_level) {
	if (img.empty())
		return;
	cv::subtract(img, cv::Scalar(black_level), img);
}

template <typename T>
inline void dpc_scalar_kernel(int r, int c, T *ptr_curr, const T *ptr_up,
							  const T *ptr_down, int threshold) {
	T center = ptr_curr[c];
	T val_L = ptr_curr[c - 2];
	T val_R = ptr_curr[c + 2];
	T val_U = ptr_up[c];
	T val_D = ptr_down[c];

	T min_val = std::min({val_L, val_R, val_U, val_D});
	T max_val = std::max({val_L, val_R, val_U, val_D});

	bool is_hot =
		(center > max_val) && ((int)center - (int)max_val > threshold);
	bool is_dead =
		(center < min_val) && ((int)min_val - (int)center > threshold);

	if (is_hot || is_dead) {
		int grad_h = std::abs((int)val_L - (int)val_R);
		int grad_v = std::abs((int)val_U - (int)val_D);

		if (grad_h < grad_v)
			ptr_curr[c] = (val_L + val_R) / 2;
		else if (grad_v < grad_h)
			ptr_curr[c] = (val_U + val_D) / 2;
		else
			ptr_curr[c] = (val_L + val_R + val_U + val_D) / 4;
	}
}

template <typename T>
void dpc_scalar_impl(cv::Mat &img, int threshold) {
	int rows = img.rows;
	int cols = img.cols;
	int border = 2;

	for (int r = border; r < rows - border; ++r) {
		T *ptr_curr = img.ptr<T>(r);
		const T *ptr_up = img.ptr<T>(r - 2);
		const T *ptr_down = img.ptr<T>(r + 2);

		for (int c = border; c < cols - border; ++c) {
			dpc_scalar_kernel(r, c, ptr_curr, ptr_up, ptr_down, threshold);
		}
	}
}

template <typename T>
void awb_scalar_impl(cv::Mat &img, const std::vector<float> &gains) {
	int rows = img.rows;
	int cols = img.cols;
	float g_tl = gains[0];
	float g_tr = gains[1];
	float g_bl = gains[2];
	float g_br = gains[3];
	const float max_val = static_cast<float>(std::numeric_limits<T>::max());

	for (int r = 0; r < rows; ++r) {
		T *ptr = img.ptr<T>(r);
		for (int c = 0; c < cols; ++c) {
			float gain;
			if (r % 2 == 0) {
				gain = (c % 2 == 0) ? g_tl : g_tr;
			} else {
				gain = (c % 2 == 0) ? g_bl : g_br;
			}
			float val = static_cast<float>(ptr[c]) * gain;
			if (val > max_val)
				val = max_val;
			ptr[c] = static_cast<T>(val);
		}
	}
}

} // namespace

// ============================================================================
// LFIsp 类实现
// ============================================================================

LFIsp::LFIsp() { cv::setNumThreads(cv::getNumberOfCPUs()); }

LFIsp::LFIsp(const IspConfig &config, const cv::Mat &lfp_img,
			 const cv::Mat &wht_img) {
	cv::setNumThreads(cv::getNumberOfCPUs());
	set_lf_img(lfp_img);
	set_white_img(wht_img);
	set_config(config);
}

LFIsp::LFIsp(const json &json_config, const cv::Mat &lfp_img,
			 const cv::Mat &wht_img) {
	cv::setNumThreads(cv::getNumberOfCPUs());
	set_lf_img(lfp_img);
	set_white_img(wht_img);
	set_config(json_config);
}

LFIsp &LFIsp::set_config(const IspConfig &new_config) {
	config_ = new_config;
	prepare_ccm_fixed_point();
	return *this;
}

LFIsp &LFIsp::set_config(const json &json_settings) {
	IspConfig new_config;

	if (json_settings.contains("bay")) {
		std::string bay_str = json_settings["bay"].get<std::string>();
		if (bay_str == "GRBG")
			new_config.bayer = BayerPattern::GRBG;
		else if (bay_str == "RGGB")
			new_config.bayer = BayerPattern::RGGB;
		else if (bay_str == "GBRG")
			new_config.bayer = BayerPattern::GBRG;
		else if (bay_str == "BGGR")
			new_config.bayer = BayerPattern::BGGR;
	}

	if (json_settings.contains("bit")) {
		new_config.bitDepth = json_settings["bit"].get<int>();
	}

	if (json_settings.contains("blc")) {
		const auto &blc = json_settings["blc"];
		if (blc.contains("black") && blc["black"].is_array()
			&& !blc["black"].empty()) {
			new_config.black_level = blc["black"][0].get<int>();
		}
		if (blc.contains("white") && blc["white"].is_array()
			&& !blc["white"].empty()) {
			new_config.white_level = blc["white"][0].get<int>();
		}
	}

	if (json_settings.contains("awb") && json_settings["awb"].is_array()) {
		new_config.awb_gains = json_settings["awb"].get<std::vector<float>>();
	}

	if (json_settings.contains("ccm") && json_settings["ccm"].is_array()) {
		new_config.ccm_matrix = json_settings["ccm"].get<std::vector<float>>();
	}

	if (json_settings.contains("gam")) {
		new_config.gamma = json_settings["gam"].get<float>();
	}

	return set_config(new_config);
}

LFIsp &LFIsp::print_config() {
	std::cout << "\n================ [LFIsp Config] ================"
			  << std::endl;
	std::cout << "Bayer Pattern : " << bayer_to_string(config_.bayer)
			  << std::endl;
	std::cout << "Bit Depth     : " << config_.bitDepth << "-bit" << std::endl;
	std::cout << "Black Level   : " << config_.black_level << std::endl;
	std::cout << "White Level   : " << config_.white_level << std::endl;

	std::cout << "AWB Gains     : [ ";
	std::cout << std::fixed << std::setprecision(3);
	if (config_.awb_gains.size() >= 4) {
		std::cout << "Gr:" << config_.awb_gains[0] << ", ";
		std::cout << "R :" << config_.awb_gains[1] << ", ";
		std::cout << "B :" << config_.awb_gains[2] << ", ";
		std::cout << "Gb:" << config_.awb_gains[3];
	} else {
		for (float g : config_.awb_gains) std::cout << g << " ";
	}
	std::cout << " ]" << std::endl;

	std::cout << "Gamma         : " << config_.gamma << std::endl;

	std::cout << "CCM Matrix    :" << std::endl;
	if (config_.ccm_matrix.size() == 9) {
		for (int i = 0; i < 3; ++i) {
			std::cout << "                [ ";
			for (int j = 0; j < 3; ++j) {
				float val = config_.ccm_matrix[i * 3 + j];
				std::cout << std::showpos << std::setw(8) << val << " ";
			}
			std::cout << std::noshowpos << "]" << std::endl;
		}
	} else {
		std::cout << "                (Invalid size: "
				  << config_.ccm_matrix.size() << ")" << std::endl;
	}
	std::cout << "================================================"
			  << std::endl;
	std::cout << std::defaultfloat;
	return *this;
}

LFIsp &LFIsp::updateImage(const cv::Mat &new_image, bool isWhite) {
	if (new_image.empty()) {
		throw std::runtime_error("Input image is empty.");
	}

	if (isWhite) {
		generate_lsc_maps(new_image);
	} else {
		lfp_img_ = new_image;
	}
	return *this;
}

LFIsp &LFIsp::set_lf_img(const cv::Mat &img) {
	if (img.empty())
		throw std::runtime_error("LF image is empty.");
	lfp_img_ = img;
	return *this;
}

LFIsp &LFIsp::set_white_img(const cv::Mat &img) {
	if (img.empty())
		throw std::runtime_error("White image is empty.");
	generate_lsc_maps(img);
	return *this;
}

std::string LFIsp::bayer_to_string(BayerPattern p) const {
	switch (p) {
		case BayerPattern::GRBG:
			return "GRBG";
		case BayerPattern::RGGB:
			return "RGGB";
		case BayerPattern::GBRG:
			return "GBRG";
		case BayerPattern::BGGR:
			return "BGGR";
		default:
			return "Unknown";
	}
}

// ============================================================================
// 标量处理流程 (Scalar Implementation)
// ============================================================================

LFIsp &LFIsp::blc() {
	if (lfp_img_.empty())
		return *this;
	if (config_.bitDepth > 8) {
		blc_scalar_impl<uint16_t>(lfp_img_, config_.black_level);
	} else {
		blc_scalar_impl<uint8_t>(lfp_img_, config_.black_level);
	}
	return *this;
}

LFIsp &LFIsp::dpc(int threshold) {
	if (lfp_img_.empty())
		return *this;
	if (config_.bitDepth > 8) {
		dpc_scalar_impl<uint16_t>(lfp_img_, threshold);
	} else {
		dpc_scalar_impl<uint8_t>(lfp_img_, threshold);
	}
	return *this;
}

LFIsp &LFIsp::lsc() {
	if (lfp_img_.empty() || lsc_gain_map_.empty())
		return *this;
	if (lfp_img_.size() != lsc_gain_map_.size())
		return *this;

	cv::Mat float_img;
	lfp_img_.convertTo(float_img, CV_32F);
	cv::multiply(float_img, lsc_gain_map_, float_img);
	float_img.convertTo(lfp_img_, lfp_img_.type());
	return *this;
}

LFIsp &LFIsp::awb() {
	if (lfp_img_.empty())
		return *this;
	if (config_.bitDepth > 8) {
		awb_scalar_impl<uint16_t>(lfp_img_, config_.awb_gains);
	} else {
		awb_scalar_impl<uint8_t>(lfp_img_, config_.awb_gains);
	}
	return *this;
}

LFIsp &LFIsp::raw_process() {
	if (lfp_img_.empty())
		return *this;
	blc().dpc().lsc().awb();
	return *this;
}

LFIsp &LFIsp::demosaic() {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.channels() != 1)
		return *this;

	int code = get_demosaic_code(config_.bayer, false);
	cv::demosaicing(lfp_img_, lfp_img_, code);
	return *this;
}

LFIsp &LFIsp::ccm() {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.channels() != 3)
		return *this;

	cv::Mat m(3, 3, CV_32F, (void *)config_.ccm_matrix.data());
	cv::transform(lfp_img_, lfp_img_, m);
	return *this;
}

LFIsp &LFIsp::gc() { return *this; }

// ============================================================================
// 快速处理流程 (SIMD Implementation Dispatcher)
// ============================================================================

LFIsp &LFIsp::blc_fast() {
	if (lfp_img_.empty())
		return *this;
	if (config_.bitDepth > 8)
		return blc_simd_u16(lfp_img_);
	else
		return blc_simd_u8(lfp_img_);
}

LFIsp &LFIsp::dpc_fast(int threshold) {
	if (lfp_img_.empty())
		return *this;
	if (config_.bitDepth > 8) {
		return dpc_simd_u16(lfp_img_, threshold << (config_.bitDepth - 10));
	} else {
		return dpc_simd_u8(lfp_img_, std::max(1, threshold >> 2));
	}
}

LFIsp &LFIsp::lsc_fast() {
	if (lfp_img_.empty())
		return *this;

	if (lsc_gain_map_int_.empty()
		|| lsc_gain_map_int_.size() != lfp_img_.size()) {
		if (lsc_gain_map_.empty())
			return *this;
		lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0f);
	}

	if (config_.bitDepth > 8)
		return lsc_simd_u16(lfp_img_);
	else
		return lsc_simd_u8(lfp_img_);
}

LFIsp &LFIsp::awb_fast() {
	if (lfp_img_.empty())
		return *this;
	if (config_.bitDepth > 8)
		return awb_simd_u16(lfp_img_);
	else
		return awb_simd_u8(lfp_img_);
}

LFIsp &LFIsp::lsc_awb_fused_fast() {
	if (lfp_img_.empty())
		return *this;

	if (lsc_gain_map_int_.empty()
		|| lsc_gain_map_int_.size() != lfp_img_.size()) {
		if (lsc_gain_map_.empty()) {
			lsc_gain_map_ = cv::Mat::ones(lfp_img_.size(), CV_32F);
		}
		lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0);
	}

	if (config_.bitDepth > 8)
		return lsc_awb_simd_u16(lfp_img_);
	else
		return lsc_awb_simd_u8(lfp_img_);
}

LFIsp &LFIsp::raw_process_fast() {
	if (lfp_img_.empty())
		return *this;
	blc_fast().dpc_fast().lsc_fast().awb_fast();
	return *this;
}

LFIsp &LFIsp::ccm_fast() {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.channels() != 3)
		return *this;

	if (ccm_matrix_int_.empty())
		prepare_ccm_fixed_point();

	if (config_.bitDepth > 8)
		return ccm_fixed_u16(lfp_img_);
	else
		return ccm_fixed_u8(lfp_img_);
}

LFIsp &LFIsp::preview(float exposure) {
	if (lfp_img_.empty())
		return *this;

	int rows = lfp_img_.rows;
	int cols = lfp_img_.cols;
	cv::Mat raw_8u(rows, cols, CV_8UC1);

	if (lsc_gain_map_int_.empty()) {
		if (!lsc_gain_map_.empty())
			lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0);
		else
			lsc_gain_map_int_.release();
	}

	// 根据位深分发
	if (config_.bitDepth > 8) {
		raw_to_8bit_with_gains_simd_u16(raw_8u, exposure);
	} else {
		raw_to_8bit_with_gains_simd_u8(raw_8u, exposure);
	}

	int code = get_demosaic_code(config_.bayer, false);
	cv::demosaicing(raw_8u, preview_img_, code);

	return *this;
}

// ============================================================================
// SIMD 具体实现
// ============================================================================

LFIsp &LFIsp::blc_simd_u16(cv::Mat &img) {
	uint16_t bl_val = static_cast<uint16_t>(config_.black_level);
	__m256i v_bl = _mm256_set1_epi16(bl_val);

	int rows = img.rows;
	int cols = img.cols;
	if (img.isContinuous()) {
		cols = rows * cols;
		rows = 1;
	}

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr = img.ptr<uint16_t>(r);
		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr + c));
			v_src = _mm256_subs_epu16(v_src, v_bl);
			_mm256_storeu_si256((__m256i *)(ptr + c), v_src);
		}
		for (; c < cols; ++c) {
			int val = ptr[c];
			ptr[c] = (val > bl_val) ? (val - bl_val) : 0;
		}
	}
	return *this;
}

LFIsp &LFIsp::blc_simd_u8(cv::Mat &img) {
	uint8_t bl_val = static_cast<uint8_t>(config_.black_level);
	__m256i v_bl = _mm256_set1_epi8(bl_val);

	int rows = img.rows;
	int cols = img.cols;
	if (img.isContinuous()) {
		cols = rows * cols;
		rows = 1;
	}

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr = img.ptr<uint8_t>(r);
		int c = 0;
		for (; c <= cols - 32; c += 32) {
			__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr + c));
			v_src = _mm256_subs_epu8(v_src, v_bl);
			_mm256_storeu_si256((__m256i *)(ptr + c), v_src);
		}
		for (; c < cols; ++c) {
			int val = ptr[c];
			ptr[c] = (val > bl_val) ? (val - bl_val) : 0;
		}
	}
	return *this;
}

LFIsp &LFIsp::dpc_simd_u16(cv::Mat &img, int threshold) {
	int rows = img.rows;
	int cols = img.cols;
	int border = 2;
	__m256i v_sign_bit = _mm256_set1_epi16((short)0x8000);
	__m256i v_thresh = _mm256_set1_epi16((short)threshold);

#pragma omp parallel for
	for (int r = border; r < rows - border; ++r) {
		uint16_t *ptr_curr = img.ptr<uint16_t>(r);
		const uint16_t *ptr_up = img.ptr<uint16_t>(r - 2);
		const uint16_t *ptr_down = img.ptr<uint16_t>(r + 2);

		int c = border;
		for (; c <= cols - border - 16; c += 16) {
			__m256i v_curr =
				_mm256_loadu_si256((const __m256i *)(ptr_curr + c));
			__m256i v_L =
				_mm256_loadu_si256((const __m256i *)(ptr_curr + c - 2));
			__m256i v_R =
				_mm256_loadu_si256((const __m256i *)(ptr_curr + c + 2));
			__m256i v_U = _mm256_loadu_si256((const __m256i *)(ptr_up + c));
			__m256i v_D = _mm256_loadu_si256((const __m256i *)(ptr_down + c));

			__m256i v_min = _mm256_min_epu16(v_L, v_R);
			v_min = _mm256_min_epu16(v_min, v_U);
			v_min = _mm256_min_epu16(v_min, v_D);

			__m256i v_max = _mm256_max_epu16(v_L, v_R);
			v_max = _mm256_max_epu16(v_max, v_U);
			v_max = _mm256_max_epu16(v_max, v_D);

			__m256i v_curr_minus_th = _mm256_subs_epu16(v_curr, v_thresh);
			__m256i v_curr_plus_th = _mm256_adds_epu16(v_curr, v_thresh);

			__m256i v_cmp_hot_lhs =
				_mm256_xor_si256(v_curr_minus_th, v_sign_bit);
			__m256i v_cmp_hot_rhs = _mm256_xor_si256(v_max, v_sign_bit);
			__m256i mask_hot = _mm256_cmpgt_epi16(v_cmp_hot_lhs, v_cmp_hot_rhs);

			__m256i v_cmp_dead_lhs = _mm256_xor_si256(v_min, v_sign_bit);
			__m256i v_cmp_dead_rhs =
				_mm256_xor_si256(v_curr_plus_th, v_sign_bit);
			__m256i mask_dead =
				_mm256_cmpgt_epi16(v_cmp_dead_lhs, v_cmp_dead_rhs);

			__m256i mask_bad = _mm256_or_si256(mask_hot, mask_dead);

			if (_mm256_testz_si256(mask_bad, mask_bad))
				continue;

			__m256i grad_h = _mm256_subs_epu16(_mm256_max_epu16(v_L, v_R),
											   _mm256_min_epu16(v_L, v_R));
			__m256i grad_v = _mm256_subs_epu16(_mm256_max_epu16(v_U, v_D),
											   _mm256_min_epu16(v_U, v_D));

			__m256i fix_h = _mm256_avg_epu16(v_L, v_R);
			__m256i fix_v = _mm256_avg_epu16(v_U, v_D);
			__m256i fix_all = _mm256_avg_epu16(fix_h, fix_v);

			__m256i v_gh_sign = _mm256_xor_si256(grad_h, v_sign_bit);
			__m256i v_gv_sign = _mm256_xor_si256(grad_v, v_sign_bit);

			__m256i mask_use_h = _mm256_cmpgt_epi16(v_gv_sign, v_gh_sign);
			__m256i mask_use_v = _mm256_cmpgt_epi16(v_gh_sign, v_gv_sign);

			__m256i v_fixed = fix_all;
			v_fixed = _mm256_blendv_epi8(v_fixed, fix_h, mask_use_h);
			v_fixed = _mm256_blendv_epi8(v_fixed, fix_v, mask_use_v);

			__m256i v_result = _mm256_blendv_epi8(v_curr, v_fixed, mask_bad);
			_mm256_storeu_si256((__m256i *)(ptr_curr + c), v_result);
		}

		for (; c < cols - border; ++c) {
			dpc_scalar_kernel(r, c, ptr_curr, ptr_up, ptr_down, threshold);
		}
	}
	return *this;
}

LFIsp &LFIsp::dpc_simd_u8(cv::Mat &img, int threshold) {
	int rows = img.rows;
	int cols = img.cols;
	int border = 2;
	__m256i v_sign_bit = _mm256_set1_epi16((short)0x8000);
	__m256i v_thresh = _mm256_set1_epi16((short)threshold);

#pragma omp parallel for
	for (int r = border; r < rows - border; ++r) {
		uint8_t *ptr_curr = img.ptr<uint8_t>(r);
		const uint8_t *ptr_up = img.ptr<uint8_t>(r - 2);
		const uint8_t *ptr_down = img.ptr<uint8_t>(r + 2);

		int c = border;
		for (; c <= cols - border - 16; c += 16) {
			__m128i v_curr_8 = _mm_loadu_si128((const __m128i *)(ptr_curr + c));
			__m128i v_L_8 =
				_mm_loadu_si128((const __m128i *)(ptr_curr + c - 2));
			__m128i v_R_8 =
				_mm_loadu_si128((const __m128i *)(ptr_curr + c + 2));
			__m128i v_U_8 = _mm_loadu_si128((const __m128i *)(ptr_up + c));
			__m128i v_D_8 = _mm_loadu_si128((const __m128i *)(ptr_down + c));

			__m256i v_curr = _mm256_cvtepu8_epi16(v_curr_8);
			__m256i v_L = _mm256_cvtepu8_epi16(v_L_8);
			__m256i v_R = _mm256_cvtepu8_epi16(v_R_8);
			__m256i v_U = _mm256_cvtepu8_epi16(v_U_8);
			__m256i v_D = _mm256_cvtepu8_epi16(v_D_8);

			__m256i v_min = _mm256_min_epu16(_mm256_min_epu16(v_L, v_R),
											 _mm256_min_epu16(v_U, v_D));
			__m256i v_max = _mm256_max_epu16(_mm256_max_epu16(v_L, v_R),
											 _mm256_max_epu16(v_U, v_D));

			__m256i v_hot = _mm256_cmpgt_epi16(
				_mm256_xor_si256(_mm256_subs_epu16(v_curr, v_thresh),
								 v_sign_bit),
				_mm256_xor_si256(v_max, v_sign_bit));
			__m256i v_dead = _mm256_cmpgt_epi16(
				_mm256_xor_si256(v_min, v_sign_bit),
				_mm256_xor_si256(_mm256_adds_epu16(v_curr, v_thresh),
								 v_sign_bit));
			__m256i mask_bad = _mm256_or_si256(v_hot, v_dead);

			if (_mm256_testz_si256(mask_bad, mask_bad))
				continue;

			__m256i g_h = _mm256_subs_epu16(_mm256_max_epu16(v_L, v_R),
											_mm256_min_epu16(v_L, v_R));
			__m256i g_v = _mm256_subs_epu16(_mm256_max_epu16(v_U, v_D),
											_mm256_min_epu16(v_U, v_D));

			__m256i fix_h = _mm256_avg_epu16(v_L, v_R);
			__m256i fix_v = _mm256_avg_epu16(v_U, v_D);
			__m256i fix_all = _mm256_avg_epu16(fix_h, fix_v);

			__m256i use_h =
				_mm256_cmpgt_epi16(_mm256_xor_si256(g_v, v_sign_bit),
								   _mm256_xor_si256(g_h, v_sign_bit));
			__m256i use_v =
				_mm256_cmpgt_epi16(_mm256_xor_si256(g_h, v_sign_bit),
								   _mm256_xor_si256(g_v, v_sign_bit));

			__m256i v_fixed = fix_all;
			v_fixed = _mm256_blendv_epi8(v_fixed, fix_h, use_h);
			v_fixed = _mm256_blendv_epi8(v_fixed, fix_v, use_v);

			__m256i v_res_16 = _mm256_blendv_epi8(v_curr, v_fixed, mask_bad);

			__m256i v_packed = _mm256_packus_epi16(v_res_16, v_res_16);
			v_packed =
				_mm256_permute4x64_epi64(v_packed, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i v_final = _mm256_castsi256_si128(v_packed);

			_mm_storeu_si128((__m128i *)(ptr_curr + c), v_final);
		}

		for (; c < cols - border; ++c) {
			dpc_scalar_kernel(r, c, ptr_curr, ptr_up, ptr_down, threshold);
		}
	}
	return *this;
}

void LFIsp::generate_lsc_maps(const cv::Mat &raw_wht) {
	int rows = raw_wht.rows;
	int cols = raw_wht.cols;

	cv::Mat float_wht;
	raw_wht.convertTo(float_wht, CV_32F);

	float bl = static_cast<float>(config_.black_level);
	cv::subtract(float_wht, cv::Scalar(bl), float_wht);
	cv::max(float_wht, 1.0f, float_wht);

	int half_h = rows / 2;
	int half_w = cols / 2;
	std::vector<cv::Mat> channels(4);
	for (int k = 0; k < 4; ++k) channels[k].create(half_h, half_w, CV_32F);

#pragma omp parallel for
	for (int r = 0; r < half_h; ++r) {
		const float *ptr_row0 = float_wht.ptr<float>(2 * r);
		const float *ptr_row1 = float_wht.ptr<float>(2 * r + 1);
		float *p0 = channels[0].ptr<float>(r);
		float *p1 = channels[1].ptr<float>(r);
		float *p2 = channels[2].ptr<float>(r);
		float *p3 = channels[3].ptr<float>(r);

		for (int c = 0; c < half_w; ++c) {
			p0[c] = ptr_row0[2 * c];
			p1[c] = ptr_row0[2 * c + 1];
			p2[c] = ptr_row1[2 * c];
			p3[c] = ptr_row1[2 * c + 1];
		}
	}

#pragma omp parallel for
	for (int k = 0; k < 4; ++k) {
		cv::GaussianBlur(channels[k], channels[k], cv::Size(7, 7), 0);
	}

	std::vector<double> maxVals(4);
	for (int k = 0; k < 4; ++k) {
		double localMax;
		cv::minMaxLoc(channels[k], nullptr, &localMax);
		if (localMax < 1e-6)
			localMax = 1.0;
		maxVals[k] = localMax;
	}

	lsc_gain_map_.create(rows, cols, CV_32F);

#pragma omp parallel for
	for (int r = 0; r < half_h; ++r) {
		float *dst0 = lsc_gain_map_.ptr<float>(2 * r);
		float *dst1 = lsc_gain_map_.ptr<float>(2 * r + 1);
		const float *p0 = channels[0].ptr<float>(r);
		const float *p1 = channels[1].ptr<float>(r);
		const float *p2 = channels[2].ptr<float>(r);
		const float *p3 = channels[3].ptr<float>(r);

		for (int c = 0; c < half_w; ++c) {
			dst0[2 * c] = (float)maxVals[0] / p0[c];
			dst0[2 * c + 1] = (float)maxVals[1] / p1[c];
			dst1[2 * c] = (float)maxVals[2] / p2[c];
			dst1[2 * c + 1] = (float)maxVals[3] / p3[c];
		}
	}
	lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0);
}

LFIsp &LFIsp::lsc_simd_u16(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr_src = img.ptr<uint16_t>(r);
		const uint16_t *ptr_gain = lsc_gain_map_int_.ptr<uint16_t>(r);
		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr_src + c));
			__m256i v_gain =
				_mm256_loadu_si256((const __m256i *)(ptr_gain + c));

			__m256i v_src_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
			__m256i v_gain_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));
			__m256i v_res_lo =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);

			__m256i v_src_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));
			__m256i v_gain_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));
			__m256i v_res_hi =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

			__m256i v_res = _mm256_packus_epi32(v_res_lo, v_res_hi);
			v_res = _mm256_permute4x64_epi64(v_res, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res);
		}
		for (; c < cols; ++c) {
			uint32_t val = (uint32_t)ptr_src[c] * ptr_gain[c];
			val >>= 12;
			if (val > 65535)
				val = 65535;
			ptr_src[c] = (uint16_t)val;
		}
	}
	return *this;
}

LFIsp &LFIsp::lsc_simd_u8(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr_src = img.ptr<uint8_t>(r);
		const uint16_t *ptr_gain = lsc_gain_map_int_.ptr<uint16_t>(r);
		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m128i v_src_small =
				_mm_loadu_si128((const __m128i *)(ptr_src + c));
			__m256i v_gain =
				_mm256_loadu_si256((const __m256i *)(ptr_gain + c));

			__m256i v_src_lo = _mm256_cvtepu8_epi32(v_src_small);
			__m256i v_gain_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));
			__m256i v_res_lo =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);

			__m128i v_src_hi_small =
				_mm_unpackhi_epi64(v_src_small, v_src_small);
			__m256i v_src_hi = _mm256_cvtepu8_epi32(v_src_hi_small);
			__m256i v_gain_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));
			__m256i v_res_hi =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

			__m256i v_packed_16 = _mm256_packus_epi32(v_res_lo, v_res_hi);
			v_packed_16 =
				_mm256_permute4x64_epi64(v_packed_16, _MM_SHUFFLE(3, 1, 2, 0));

			__m128i v_packed_u8 =
				_mm_packus_epi16(_mm256_castsi256_si128(v_packed_16),
								 _mm256_extracti128_si256(v_packed_16, 1));
			_mm_storeu_si128((__m128i *)(ptr_src + c), v_packed_u8);
		}
		for (; c < cols; ++c) {
			uint32_t val = (uint32_t)ptr_src[c] * ptr_gain[c];
			val >>= 12;
			if (val > 255)
				val = 255;
			ptr_src[c] = (uint8_t)val;
		}
	}
	return *this;
}

LFIsp &LFIsp::awb_simd_u16(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = 4096.0f;
	uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
	uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
	uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
	uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_gain_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_gain_row1 = _mm256_set1_epi32(p_row1);

#pragma omp parallel for
	for (int r = 0; r < rows; r += 2) {
		if (r + 1 >= rows)
			continue;
		uint16_t *ptr0 = img.ptr<uint16_t>(r);
		uint16_t *ptr1 = img.ptr<uint16_t>(r + 1);
		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v0 = _mm256_loadu_si256((const __m256i *)(ptr0 + c));
			__m256i v0_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v0));
			__m256i v0_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v0, 1));
			__m256i vg0_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain_row0));
			__m256i vg0_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain_row0, 1));
			v0_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v0_lo, vg0_lo), 12);
			v0_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v0_hi, vg0_hi), 12);
			v0 = _mm256_packus_epi32(v0_lo, v0_hi);
			v0 = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr0 + c), v0);

			__m256i v1 = _mm256_loadu_si256((const __m256i *)(ptr1 + c));
			__m256i v1_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v1));
			__m256i v1_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v1, 1));
			__m256i vg1_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain_row1));
			__m256i vg1_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain_row1, 1));
			v1_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v1_lo, vg1_lo), 12);
			v1_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v1_hi, vg1_hi), 12);
			v1 = _mm256_packus_epi32(v1_lo, v1_hi);
			v1 = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr1 + c), v1);
		}

		for (; c < cols; ++c) {
			uint32_t val0 = (uint32_t)ptr0[c] * ((c % 2) ? g_tr : g_tl);
			val0 >>= 12;
			ptr0[c] = (val0 > 65535) ? 65535 : (uint16_t)val0;
			uint32_t val1 = (uint32_t)ptr1[c] * ((c % 2) ? g_br : g_bl);
			val1 >>= 12;
			ptr1[c] = (val1 > 65535) ? 65535 : (uint16_t)val1;
		}
	}
	return *this;
}

LFIsp &LFIsp::awb_simd_u8(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = 4096.0f;
	uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
	uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
	uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
	uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_gain_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_gain_row1 = _mm256_set1_epi32(p_row1);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr_src = img.ptr<uint8_t>(r);
		__m256i v_gain = (r % 2 == 0) ? v_gain_row0 : v_gain_row1;

		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m128i v_small = _mm_loadu_si128((const __m128i *)(ptr_src + c));
			__m256i v_src_16 = _mm256_cvtepu8_epi16(v_small);

			__m256i v_src_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src_16));
			__m256i v_gain_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));
			__m256i v_src_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src_16, 1));
			__m256i v_gain_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));

			__m256i v_res_lo =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);
			__m256i v_res_hi =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

			__m256i v_packed_16 = _mm256_packus_epi32(v_res_lo, v_res_hi);
			v_packed_16 =
				_mm256_permute4x64_epi64(v_packed_16, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i v_final =
				_mm_packus_epi16(_mm256_castsi256_si128(v_packed_16),
								 _mm256_extracti128_si256(v_packed_16, 1));
			_mm_storeu_si128((__m128i *)(ptr_src + c), v_final);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint16_t gain = (c % 2 == 0) ? awb_0 : awb_1;
			uint32_t val = ((uint32_t)ptr_src[c] * gain) >> 12;
			if (val > 255)
				val = 255;
			ptr_src[c] = (uint8_t)val;
		}
	}
	return *this;
}

LFIsp &LFIsp::raw_to_8bit_with_gains_simd_u16(cv::Mat &dst_8u, float exposure) {
	int rows = lfp_img_.rows;
	int cols = lfp_img_.cols;

	uint16_t bl_val = static_cast<uint16_t>(config_.black_level);
	__m256i v_bl = _mm256_set1_epi16(bl_val);

	float effective_range =
		static_cast<float>(config_.white_level - config_.black_level);
	if (effective_range < 1.0f)
		effective_range = 1.0f;
	float total_scale_factor = (255.0f / effective_range) * exposure * 4096.0f;

	auto calc_gain = [&](float awb_g) -> uint16_t {
		float val = awb_g * total_scale_factor;
		if (val > 65535.0f)
			val = 65535.0f;
		return static_cast<uint16_t>(val);
	};

	uint16_t g_tl = calc_gain(config_.awb_gains[0]);
	uint16_t g_tr = calc_gain(config_.awb_gains[1]);
	uint16_t g_bl = calc_gain(config_.awb_gains[2]);
	uint16_t g_br = calc_gain(config_.awb_gains[3]);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

	bool has_lsc = !lsc_gain_map_int_.empty()
				   && lsc_gain_map_int_.size() == lfp_img_.size();

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		const uint16_t *src = lfp_img_.ptr<uint16_t>(r);
		const uint16_t *lsc =
			has_lsc ? lsc_gain_map_int_.ptr<uint16_t>(r) : nullptr;
		uint8_t *dst = dst_8u.ptr<uint8_t>(r);
		__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v_src = _mm256_loadu_si256((const __m256i *)(src + c));
			v_src = _mm256_subs_epu16(v_src, v_bl);

			if (has_lsc) {
				__m256i v_lsc = _mm256_loadu_si256((const __m256i *)(lsc + c));
				__m256i v_src_lo =
					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
				__m256i v_lsc_lo =
					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc));
				v_src_lo = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_src_lo, v_lsc_lo), 12);

				__m256i v_src_hi =
					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));
				__m256i v_lsc_hi =
					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc, 1));
				v_src_hi = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_src_hi, v_lsc_hi), 12);

				v_src = _mm256_packus_epi32(v_src_lo, v_src_hi);
				v_src =
					_mm256_permute4x64_epi64(v_src, _MM_SHUFFLE(3, 1, 2, 0));
			}

			__m256i v_src_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
			__m256i v_awb_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
			v_src_lo =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_awb_lo), 12);

			__m256i v_src_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));
			__m256i v_awb_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));
			v_src_hi =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_awb_hi), 12);

			__m256i v_res_16 = _mm256_packus_epi32(v_src_lo, v_src_hi);
			v_res_16 =
				_mm256_permute4x64_epi64(v_res_16, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i v_res_8 =
				_mm_packus_epi16(_mm256_castsi256_si128(v_res_16),
								 _mm256_extracti128_si256(v_res_16, 1));
			_mm_storeu_si128((__m128i *)(dst + c), v_res_8);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint32_t val = src[c];
			val = (val > bl_val) ? (val - bl_val) : 0;
			if (has_lsc)
				val = (val * lsc[c]) >> 12;
			uint32_t awb = (c % 2 == 0) ? awb_0 : awb_1;
			val = (val * awb) >> 12;
			if (val > 255)
				val = 255;
			dst[c] = static_cast<uint8_t>(val);
		}
	}
	return *this;
}

LFIsp &LFIsp::raw_to_8bit_with_gains_simd_u8(cv::Mat &dst_8u, float exposure) {
	int rows = lfp_img_.rows;
	int cols = lfp_img_.cols;

	int bl_shift = (config_.bitDepth > 8) ? (config_.bitDepth - 8) : 0;
	uint8_t bl_val = static_cast<uint8_t>(config_.black_level >> bl_shift);
	__m256i v_bl = _mm256_set1_epi8(bl_val);

	float effective_range = 255.0f - bl_val;
	if (effective_range < 1.0f)
		effective_range = 1.0f;
	float total_scale_factor = (255.0f / effective_range) * exposure * 4096.0f;

	auto calc_gain = [&](float awb_g) -> uint16_t {
		float val = awb_g * total_scale_factor;
		if (val > 65535.0f)
			val = 65535.0f;
		return static_cast<uint16_t>(val);
	};

	uint16_t g_tl = calc_gain(config_.awb_gains[0]);
	uint16_t g_tr = calc_gain(config_.awb_gains[1]);
	uint16_t g_bl = calc_gain(config_.awb_gains[2]);
	uint16_t g_br = calc_gain(config_.awb_gains[3]);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

	bool has_lsc = !lsc_gain_map_int_.empty()
				   && lsc_gain_map_int_.size() == lfp_img_.size();

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *src = lfp_img_.ptr<uint8_t>(r);
		uint8_t *dst = dst_8u.ptr<uint8_t>(r);
		const uint16_t *lsc =
			has_lsc ? lsc_gain_map_int_.ptr<uint16_t>(r) : nullptr;
		__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

		int c = 0;
		for (; c <= cols - 32; c += 32) {
			__m256i v_src_32 = _mm256_loadu_si256((const __m256i *)(src + c));
			v_src_32 = _mm256_subs_epu8(v_src_32, v_bl);

			__m256i v_lsc_0, v_lsc_1;
			if (has_lsc) {
				v_lsc_0 = _mm256_loadu_si256((const __m256i *)(lsc + c));
				v_lsc_1 = _mm256_loadu_si256((const __m256i *)(lsc + c + 16));
			} else {
				__m256i v_one = _mm256_set1_epi16(4096);
				v_lsc_0 = v_one;
				v_lsc_1 = v_one;
			}

			auto process_half = [&](__m128i v_p_8,
									__m256i v_lsc_16) -> __m256i {
				__m256i v_p_lo = _mm256_cvtepu8_epi32(v_p_8);
				__m256i v_p_hi =
					_mm256_cvtepu8_epi32(_mm_unpackhi_epi64(v_p_8, v_p_8));

				__m256i v_lsc_lo =
					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc_16));
				__m256i v_lsc_hi = _mm256_cvtepu16_epi32(
					_mm256_extracti128_si256(v_lsc_16, 1));

				__m256i v_awb_lo =
					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
				__m256i v_awb_hi =
					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

				__m256i v_gain_lo = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), 12);
				__m256i v_gain_hi = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), 12);

				v_p_lo = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_p_lo, v_gain_lo), 12);
				v_p_hi = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_p_hi, v_gain_hi), 12);

				__m256i v_res_16 = _mm256_packus_epi32(v_p_lo, v_p_hi);
				return _mm256_permute4x64_epi64(v_res_16,
												_MM_SHUFFLE(3, 1, 2, 0));
			};

			__m256i v_res_0 =
				process_half(_mm256_castsi256_si128(v_src_32), v_lsc_0);
			__m256i v_res_1 =
				process_half(_mm256_extracti128_si256(v_src_32, 1), v_lsc_1);

			__m256i v_res_u8 = _mm256_packus_epi16(v_res_0, v_res_1);
			v_res_u8 =
				_mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(dst + c), v_res_u8);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;

		for (; c < cols; ++c) {
			uint32_t val = src[c];
			val = (val > bl_val) ? (val - bl_val) : 0;
			if (has_lsc)
				val = (val * lsc[c]) >> 12;
			uint32_t awb = (c % 2 == 0) ? awb_0 : awb_1;
			val = (val * awb) >> 12;
			if (val > 255)
				val = 255;
			dst[c] = static_cast<uint8_t>(val);
		}
	}
	return *this;
}

LFIsp &LFIsp::lsc_awb_simd_u16(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = 4096.0f;
	uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
	uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
	uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
	uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr_src = img.ptr<uint16_t>(r);
		const uint16_t *ptr_lsc = lsc_gain_map_int_.ptr<uint16_t>(r);
		__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v_lsc = _mm256_loadu_si256((const __m256i *)(ptr_lsc + c));

			__m256i v_lsc_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc));
			__m256i v_lsc_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc, 1));
			__m256i v_awb_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
			__m256i v_awb_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

			__m256i v_gain_lo =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), 12);
			__m256i v_gain_hi =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), 12);

			__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr_src + c));
			__m256i v_src_lo =
				_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
			__m256i v_src_hi =
				_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));

			v_src_lo =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);
			v_src_hi =
				_mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

			__m256i v_res = _mm256_packus_epi32(v_src_lo, v_src_hi);
			v_res = _mm256_permute4x64_epi64(v_res, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint16_t lsc = ptr_lsc[c];
			uint16_t awb = (c % 2 == 0) ? awb_0 : awb_1;
			uint32_t total_gain = ((uint32_t)lsc * awb) >> 12;
			uint32_t val = ((uint32_t)ptr_src[c] * total_gain) >> 12;
			if (val > 65535)
				val = 65535;
			ptr_src[c] = static_cast<uint16_t>(val);
		}
	}
	return *this;
}

LFIsp &LFIsp::lsc_awb_simd_u8(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = 4096.0f;
	uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
	uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
	uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
	uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr_src = img.ptr<uint8_t>(r);
		const uint16_t *ptr_lsc = lsc_gain_map_int_.ptr<uint16_t>(r);
		__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

		int c = 0;
		for (; c <= cols - 32; c += 32) {
			__m256i v_src_32 =
				_mm256_loadu_si256((const __m256i *)(ptr_src + c));
			__m256i v_lsc_0 =
				_mm256_loadu_si256((const __m256i *)(ptr_lsc + c));
			__m256i v_lsc_1 =
				_mm256_loadu_si256((const __m256i *)(ptr_lsc + c + 16));

			auto process_half = [&](__m128i v_p_8,
									__m256i v_lsc_16) -> __m256i {
				__m256i v_p_lo = _mm256_cvtepu8_epi32(v_p_8);
				__m256i v_p_hi =
					_mm256_cvtepu8_epi32(_mm_unpackhi_epi64(v_p_8, v_p_8));

				__m256i v_lsc_lo =
					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc_16));
				__m256i v_lsc_hi = _mm256_cvtepu16_epi32(
					_mm256_extracti128_si256(v_lsc_16, 1));

				__m256i v_awb_lo =
					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
				__m256i v_awb_hi =
					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

				__m256i v_gain_lo = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), 12);
				__m256i v_gain_hi = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), 12);

				v_p_lo = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_p_lo, v_gain_lo), 12);
				v_p_hi = _mm256_srli_epi32(
					_mm256_mullo_epi32(v_p_hi, v_gain_hi), 12);

				__m256i v_res_16 = _mm256_packus_epi32(v_p_lo, v_p_hi);
				return _mm256_permute4x64_epi64(v_res_16,
												_MM_SHUFFLE(3, 1, 2, 0));
			};

			__m256i v_res_0 =
				process_half(_mm256_castsi256_si128(v_src_32), v_lsc_0);
			__m256i v_res_1 =
				process_half(_mm256_extracti128_si256(v_src_32, 1), v_lsc_1);

			__m256i v_res_u8 = _mm256_packus_epi16(v_res_0, v_res_1);
			v_res_u8 =
				_mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res_u8);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint16_t lsc = ptr_lsc[c];
			uint16_t awb = (c % 2 == 0) ? awb_0 : awb_1;
			uint32_t total_gain = ((uint32_t)lsc * awb) >> 12;
			uint32_t val = ((uint32_t)ptr_src[c] * total_gain) >> 12;
			if (val > 255)
				val = 255;
			ptr_src[c] = static_cast<uint8_t>(val);
		}
	}
	return *this;
}

LFIsp &LFIsp::ccm_fixed_u16(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;
	const int32_t c00 = ccm_matrix_int_[0], c01 = ccm_matrix_int_[1],
				  c02 = ccm_matrix_int_[2];
	const int32_t c10 = ccm_matrix_int_[3], c11 = ccm_matrix_int_[4],
				  c12 = ccm_matrix_int_[5];
	const int32_t c20 = ccm_matrix_int_[6], c21 = ccm_matrix_int_[7],
				  c22 = ccm_matrix_int_[8];

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr = img.ptr<uint16_t>(r);
		for (int c = 0; c < cols; ++c) {
			int idx = c * 3;
			int32_t r_val = ptr[idx + 0];
			int32_t g_val = ptr[idx + 1];
			int32_t b_val = ptr[idx + 2];
			int32_t new_ch0 = (r_val * c00 + g_val * c01 + b_val * c02) >> 12;
			int32_t new_ch1 = (r_val * c10 + g_val * c11 + b_val * c12) >> 12;
			int32_t new_ch2 = (r_val * c20 + g_val * c21 + b_val * c22) >> 12;
			if (new_ch0 < 0)
				new_ch0 = 0;
			else if (new_ch0 > 65535)
				new_ch0 = 65535;
			if (new_ch1 < 0)
				new_ch1 = 0;
			else if (new_ch1 > 65535)
				new_ch1 = 65535;
			if (new_ch2 < 0)
				new_ch2 = 0;
			else if (new_ch2 > 65535)
				new_ch2 = 65535;
			ptr[idx + 0] = static_cast<uint16_t>(new_ch0);
			ptr[idx + 1] = static_cast<uint16_t>(new_ch1);
			ptr[idx + 2] = static_cast<uint16_t>(new_ch2);
		}
	}
	return *this;
}

LFIsp &LFIsp::ccm_fixed_u8(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;
	const int32_t c00 = ccm_matrix_int_[0], c01 = ccm_matrix_int_[1],
				  c02 = ccm_matrix_int_[2];
	const int32_t c10 = ccm_matrix_int_[3], c11 = ccm_matrix_int_[4],
				  c12 = ccm_matrix_int_[5];
	const int32_t c20 = ccm_matrix_int_[6], c21 = ccm_matrix_int_[7],
				  c22 = ccm_matrix_int_[8];

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr = img.ptr<uint8_t>(r);
		for (int c = 0; c < cols; ++c) {
			int idx = c * 3;
			int32_t r_val = ptr[idx];
			int32_t g_val = ptr[idx + 1];
			int32_t b_val = ptr[idx + 2];
			int32_t new_ch0 = (r_val * c00 + g_val * c01 + b_val * c02) >> 12;
			int32_t new_ch1 = (r_val * c10 + g_val * c11 + b_val * c12) >> 12;
			int32_t new_ch2 = (r_val * c20 + g_val * c21 + b_val * c22) >> 12;
			if (new_ch0 < 0)
				new_ch0 = 0;
			else if (new_ch0 > 255)
				new_ch0 = 255;
			if (new_ch1 < 0)
				new_ch1 = 0;
			else if (new_ch1 > 255)
				new_ch1 = 255;
			if (new_ch2 < 0)
				new_ch2 = 0;
			else if (new_ch2 > 255)
				new_ch2 = 255;
			ptr[idx] = (uint8_t)new_ch0;
			ptr[idx + 1] = (uint8_t)new_ch1;
			ptr[idx + 2] = (uint8_t)new_ch2;
		}
	}
	return *this;
}

LFIsp &LFIsp::prepare_ccm_fixed_point() {
	if (config_.ccm_matrix.empty())
		return *this;
	ccm_matrix_int_.resize(9);
	const float scale = 4096.0f;
	for (int i = 0; i < 9; ++i) {
		ccm_matrix_int_[i] =
			static_cast<int32_t>(config_.ccm_matrix[i] * scale);
	}
	return *this;
}

LFIsp &LFIsp::resample(bool dehex) {
	int num_views = maps.extract.size() / 2;
	cv::Mat src = preview_img_.empty() ? lfp_img_ : preview_img_;
	sais.clear();
	sais.resize(num_views);

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_views; ++i) {
		cv::Mat temp;
		cv::remap(src, temp, maps.extract[i * 2], maps.extract[i * 2 + 1],
				  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
		if (dehex) {
			cv::remap(temp, temp, maps.dehex[0], maps.dehex[1],
					  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
		}
		sais[i] = temp;
	}
	return *this;
}

int LFIsp::get_demosaic_code(BayerPattern pattern, bool gray) {
	switch (pattern) {
		case BayerPattern::GRBG:
			return gray ? cv::COLOR_BayerGR2GRAY
						: cv::COLOR_BayerGR2RGB; // 第0行是 G, R
		case BayerPattern::RGGB:
			return gray ? cv::COLOR_BayerRG2GRAY
						: cv::COLOR_BayerRG2RGB; // 第0行是 R, G
		case BayerPattern::GBRG:
			return gray ? cv::COLOR_BayerGB2GRAY
						: cv::COLOR_BayerGB2RGB; // 第0行是 G, B
		case BayerPattern::BGGR:
			return gray ? cv::COLOR_BayerBG2GRAY
						: cv::COLOR_BayerBG2RGB; // 第0行是 B, G
		default:
			// 默认处理
			return gray ? cv::COLOR_BayerGR2GRAY : cv::COLOR_BayerGR2RGB;
	}
}

LFIsp &LFIsp::compute_lab_stats(const cv::Mat &src, cv::Scalar &mean,
								cv::Scalar &stddev) {
	cv::Mat lab;
	int depth = src.depth();
	if (depth == CV_8U) {
		src.convertTo(lab, CV_32F, 1.0 / 255.0);
	} else if (depth == CV_16U) {
		src.convertTo(lab, CV_32F, 1.0 / 65535.0);
	} else if (depth == CV_32F || depth == CV_64F) {
		lab = src;
	} else {
		src.convertTo(lab, CV_32F);
	}
	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	cv::meanStdDev(lab, mean, stddev);
	return *this;
}

LFIsp &LFIsp::apply_reinhard_transfer(cv::Mat &target,
									  const cv::Scalar &ref_mean,
									  const cv::Scalar &ref_std) {
	if (target.empty())
		return *this;
	cv::Mat lab;
	if (target.depth() == CV_8U) {
		target.convertTo(lab, CV_32F, 1.0 / 255.0);
	} else {
		target.convertTo(lab, CV_32F);
	}
	cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
	cv::Scalar src_mean, src_std;
	cv::meanStdDev(lab, src_mean, src_std);
	std::vector<cv::Mat> channels;
	cv::split(lab, channels);
	for (int i = 0; i < 3; ++i) {
		double s_std = (src_std[i] < 1e-6) ? 1e-6 : src_std[i];
		double alpha = ref_std[i] / s_std;
		double beta = ref_mean[i] - alpha * src_mean[i];
		channels[i].convertTo(channels[i], -1, alpha, beta);
	}
	cv::merge(channels, lab);
	cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);
	if (target.depth() == CV_8U) {
		lab.convertTo(target, CV_8U, 255.0);
	} else {
		lab.copyTo(target);
	}
	return *this;
}

LFIsp &LFIsp::color_equalize() {
	if (sais.empty())
		return *this;
	int num_views = sais.size();
	int side_len = static_cast<int>(std::sqrt(num_views));
	int center_idx = (side_len / 2) * side_len + (side_len / 2);
	if (center_idx >= num_views)
		center_idx = 0;
	const cv::Mat &ref_img = sais[center_idx];
	cv::Scalar ref_mean, ref_std;
	compute_lab_stats(ref_img, ref_mean, ref_std);

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_views; ++i) {
		if (i == center_idx)
			continue;
		apply_reinhard_transfer(sais[i], ref_mean, ref_std);
	}
	return *this;
}