#include "lfisp.h"

#include "colormatcher.h"
#include "hexgrid_fit.h"
#include "utils.h"

#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

extern void launch_dpc_8u_inplace(cv::cuda::GpuMat &img, int threshold, cv::cuda::Stream &stream);
extern void launch_nr_8u(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float sigma_s, float sigma_r,
						 cv::cuda::Stream &stream);
extern void launch_lsc_8u_apply_32f(cv::cuda::GpuMat &img, const cv::cuda::GpuMat &lsc_map, float exposure,
									cv::cuda::Stream &stream);
extern void launch_awb_8u(cv::cuda::GpuMat &img, float g00, float g01, float g10, float g11, cv::cuda::Stream &stream);
extern void launch_fused_lsc_awb(cv::cuda::GpuMat &img, const cv::cuda::GpuMat &lsc_map, float exposure, float g00,
								 float g01, float g10, float g11, cv::cuda::Stream &stream);
extern void launch_ccm_8uc3(cv::cuda::GpuMat &img, const float *m, cv::cuda::Stream &stream);
extern void launch_gc_8u(cv::cuda::GpuMat &img, float gamma, cv::cuda::Stream &stream);
extern void launch_ccm_gamma_fused(cv::cuda::GpuMat &img, const float *m, float gamma, cv::cuda::Stream &stream);
extern void launch_se_gpu(cv::cuda::GpuMat &img, float factor, cv::cuda::Stream &stream);
extern void launch_uvnr_8uc3(cv::cuda::GpuMat &img, float sigma_s, float sigma_r, cv::cuda::Stream &stream);

namespace {

template <typename T>
inline void dpc_scalar_kernel(int r, int c, T *ptr_curr, const T *ptr_up, const T *ptr_down, int threshold) {
	T center = ptr_curr[c];
	T val_L = ptr_curr[c - 2];
	T val_R = ptr_curr[c + 2];
	T val_U = ptr_up[c];
	T val_D = ptr_down[c];

	T min_val = std::min({val_L, val_R, val_U, val_D});
	T max_val = std::max({val_L, val_R, val_U, val_D});

	bool is_hot = (center > max_val) && ((int)center - (int)max_val > threshold);
	bool is_dead = (center < min_val) && ((int)min_val - (int)center > threshold);

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

void blc_simd_u16(cv::Mat &img, int black_level) {
	uint16_t bl_val = static_cast<uint16_t>(black_level);
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
}

void blc_simd_u8(cv::Mat &img, int black_level) {
	uint8_t bl_val = static_cast<uint8_t>(black_level);
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
}

void dpc_simd_u16(cv::Mat &img, int threshold) {
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
			__m256i v_curr = _mm256_loadu_si256((const __m256i *)(ptr_curr + c));
			__m256i v_L = _mm256_loadu_si256((const __m256i *)(ptr_curr + c - 2));
			__m256i v_R = _mm256_loadu_si256((const __m256i *)(ptr_curr + c + 2));
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

			__m256i v_cmp_hot_lhs = _mm256_xor_si256(v_curr_minus_th, v_sign_bit);
			__m256i v_cmp_hot_rhs = _mm256_xor_si256(v_max, v_sign_bit);
			__m256i mask_hot = _mm256_cmpgt_epi16(v_cmp_hot_lhs, v_cmp_hot_rhs);

			__m256i v_cmp_dead_lhs = _mm256_xor_si256(v_min, v_sign_bit);
			__m256i v_cmp_dead_rhs = _mm256_xor_si256(v_curr_plus_th, v_sign_bit);
			__m256i mask_dead = _mm256_cmpgt_epi16(v_cmp_dead_lhs, v_cmp_dead_rhs);

			__m256i mask_bad = _mm256_or_si256(mask_hot, mask_dead);

			if (_mm256_testz_si256(mask_bad, mask_bad))
				continue;

			__m256i grad_h = _mm256_subs_epu16(_mm256_max_epu16(v_L, v_R), _mm256_min_epu16(v_L, v_R));
			__m256i grad_v = _mm256_subs_epu16(_mm256_max_epu16(v_U, v_D), _mm256_min_epu16(v_U, v_D));

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
}

void dpc_simd_u8(cv::Mat &img, int threshold) {
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
			__m128i v_L_8 = _mm_loadu_si128((const __m128i *)(ptr_curr + c - 2));
			__m128i v_R_8 = _mm_loadu_si128((const __m128i *)(ptr_curr + c + 2));
			__m128i v_U_8 = _mm_loadu_si128((const __m128i *)(ptr_up + c));
			__m128i v_D_8 = _mm_loadu_si128((const __m128i *)(ptr_down + c));

			__m256i v_curr = _mm256_cvtepu8_epi16(v_curr_8);
			__m256i v_L = _mm256_cvtepu8_epi16(v_L_8);
			__m256i v_R = _mm256_cvtepu8_epi16(v_R_8);
			__m256i v_U = _mm256_cvtepu8_epi16(v_U_8);
			__m256i v_D = _mm256_cvtepu8_epi16(v_D_8);

			__m256i v_min = _mm256_min_epu16(_mm256_min_epu16(v_L, v_R), _mm256_min_epu16(v_U, v_D));
			__m256i v_max = _mm256_max_epu16(_mm256_max_epu16(v_L, v_R), _mm256_max_epu16(v_U, v_D));

			__m256i v_hot = _mm256_cmpgt_epi16(_mm256_xor_si256(_mm256_subs_epu16(v_curr, v_thresh), v_sign_bit),
											   _mm256_xor_si256(v_max, v_sign_bit));
			__m256i v_dead = _mm256_cmpgt_epi16(_mm256_xor_si256(v_min, v_sign_bit),
												_mm256_xor_si256(_mm256_adds_epu16(v_curr, v_thresh), v_sign_bit));
			__m256i mask_bad = _mm256_or_si256(v_hot, v_dead);

			if (_mm256_testz_si256(mask_bad, mask_bad))
				continue;

			__m256i g_h = _mm256_subs_epu16(_mm256_max_epu16(v_L, v_R), _mm256_min_epu16(v_L, v_R));
			__m256i g_v = _mm256_subs_epu16(_mm256_max_epu16(v_U, v_D), _mm256_min_epu16(v_U, v_D));

			__m256i fix_h = _mm256_avg_epu16(v_L, v_R);
			__m256i fix_v = _mm256_avg_epu16(v_U, v_D);
			__m256i fix_all = _mm256_avg_epu16(fix_h, fix_v);

			__m256i use_h = _mm256_cmpgt_epi16(_mm256_xor_si256(g_v, v_sign_bit), _mm256_xor_si256(g_h, v_sign_bit));
			__m256i use_v = _mm256_cmpgt_epi16(_mm256_xor_si256(g_h, v_sign_bit), _mm256_xor_si256(g_v, v_sign_bit));

			__m256i v_fixed = fix_all;
			v_fixed = _mm256_blendv_epi8(v_fixed, fix_h, use_h);
			v_fixed = _mm256_blendv_epi8(v_fixed, fix_v, use_v);

			__m256i v_res_16 = _mm256_blendv_epi8(v_curr, v_fixed, mask_bad);

			__m256i v_packed = _mm256_packus_epi16(v_res_16, v_res_16);
			v_packed = _mm256_permute4x64_epi64(v_packed, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i v_final = _mm256_castsi256_si128(v_packed);

			_mm_storeu_si128((__m128i *)(ptr_curr + c), v_final);
		}

		for (; c < cols - border; ++c) {
			dpc_scalar_kernel(r, c, ptr_curr, ptr_up, ptr_down, threshold);
		}
	}
}

std::vector<float> prepare_range_lut_generic(float sigma_color, int max_val) {
	std::vector<float> lut(max_val + 1);
	float inv_2_sigma2 = -1.0f / (2.0f * sigma_color * sigma_color);
	for (int i = 0; i <= max_val; ++i) {
		lut[i] = std::exp(i * i * inv_2_sigma2);
	}
	return lut;
}

void nr_simd_u16(cv::Mat &img, float sigma_spatial, float sigma_color) {
	const int rows = img.rows;
	const int cols = img.cols;
	cv::Mat src = img.clone();

	int radius = static_cast<int>(sigma_spatial * 1.5f);
	if (radius < 1)
		radius = 1;

	auto range_lut = prepare_range_lut_generic(sigma_color, 65535);
	const float *lut_ptr = range_lut.data();

	// 空间权重预计算
	std::vector<float> sw((radius * 2 + 1) * (radius * 2 + 1));
	float inv_2_s2 = -1.0f / (2.0f * sigma_spatial * sigma_spatial);
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			sw[(i + radius) * (radius * 2 + 1) + (j + radius)] =
				std::exp(static_cast<float>(i * i + j * j) * 4.0f * inv_2_s2);
		}
	}

#pragma omp parallel for
	for (int r = radius * 2; r < rows - radius * 2; ++r) {
		uint16_t *dst = img.ptr<uint16_t>(r);
		for (int c = radius * 2; c < cols - radius * 2; ++c) {
			float sw_sum = 0, sv_sum = 0;
			uint16_t center = src.ptr<uint16_t>(r)[c];

			for (int i = -radius; i <= radius; ++i) {
				const uint16_t *row = src.ptr<uint16_t>(r + i * 2);
				for (int j = -radius; j <= radius; ++j) {
					// 同色采样核心修复
					uint16_t val = row[c + j * 2];
					float w = lut_ptr[std::abs(center - val)] * sw[(i + radius) * (radius * 2 + 1) + (j + radius)];
					sw_sum += w;
					sv_sum += w * val;
				}
			}
			dst[c] = static_cast<uint16_t>(sv_sum / sw_sum + 0.5f);
		}
	}
}

void nr_simd_u8(cv::Mat &img, float sigma_spatial, float sigma_color) {
	const int rows = img.rows;
	const int cols = img.cols;
	cv::Mat src = img.clone();

	// 1. 自动确定搜索半径 (匹配 GPU: radius = sigma_s * 1.5)
	int radius = static_cast<int>(sigma_spatial * 1.5f);
	if (radius < 1)
		radius = 1;

	// 2. 预计算查找表
	auto range_lut = prepare_range_lut_generic(sigma_color, 255);
	const float *lut_ptr = range_lut.data();

	// 预计算空间权重：d2 = (i*i + j*j) * 4.0f (因为步长为 2)
	std::vector<float> space_weights((radius * 2 + 1) * (radius * 2 + 1));
	float inv_2sigma_s2 = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			float d2 = static_cast<float>(i * i + j * j) * 4.0f;
			space_weights[(i + radius) * (radius * 2 + 1) + (j + radius)] = std::exp(-(d2 * inv_2sigma_s2));
		}
	}

	// 3. 并行行扫描
#pragma omp parallel for
	for (int r = radius * 2; r < rows - radius * 2; ++r) {
		uint8_t *dst_ptr = img.ptr<uint8_t>(r);
		int c = radius * 2;

		// SIMD 主循环：每次处理 16 个像素
		for (; c <= cols - radius * 2 - 16; c += 16) {
			// 加载中心像素块 (包含 R-G 或 G-B 混合模式)
			__m128i v_center_8 = _mm_loadu_si128((const __m128i *)(src.ptr<uint8_t>(r) + c));
			__m256i v_center_16 = _mm256_cvtepu8_epi16(v_center_8);

			// 累加器 (float)
			__m256 sum_v_lo = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_center_16)));
			__m256 sum_v_hi = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_center_16, 1)));
			__m256 sum_w_lo = _mm256_set1_ps(1.0f);
			__m256 sum_w_hi = _mm256_set1_ps(1.0f);

			for (int i = -radius; i <= radius; ++i) {
				const uint8_t *src_row_ptr = src.ptr<uint8_t>(r + i * 2); // 步长 2
				for (int j = -radius; j <= radius; ++j) {
					if (i == 0 && j == 0)
						continue;

					// 【核心修复】采样同色邻域：偏移必须为 j * 2
					__m128i v_nb_8 = _mm_loadu_si128((const __m128i *)(src_row_ptr + c + j * 2));
					__m256i v_nb_16 = _mm256_cvtepu8_epi16(v_nb_8);

					// 计算亮度差值
					__m256i v_diff = _mm256_abs_epi16(_mm256_sub_epi16(v_center_16, v_nb_16));

					// SIMD Gather 查表获取 Range Weight
					__m256 v_w_r_lo =
						_mm256_i32gather_ps(lut_ptr, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_diff)), 4);
					__m256 v_w_r_hi =
						_mm256_i32gather_ps(lut_ptr, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_diff, 1)), 4);

					// 结合空间权重
					float sw = space_weights[(i + radius) * (radius * 2 + 1) + (j + radius)];
					__m256 v_w_lo = _mm256_mul_ps(v_w_r_lo, _mm256_set1_ps(sw));
					__m256 v_w_hi = _mm256_mul_ps(v_w_r_hi, _mm256_set1_ps(sw));

					sum_w_lo = _mm256_add_ps(sum_w_lo, v_w_lo);
					sum_w_hi = _mm256_add_ps(sum_w_hi, v_w_hi);

					// 累加像素值
					__m256 v_nb_f_lo = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_nb_16)));
					__m256 v_nb_f_hi = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_nb_16, 1)));
					sum_v_lo = _mm256_add_ps(sum_v_lo, _mm256_mul_ps(v_w_lo, v_nb_f_lo));
					sum_v_hi = _mm256_add_ps(sum_v_hi, _mm256_mul_ps(v_w_hi, v_nb_f_hi));
				}
			}

			// 归一化结果
			__m256i v_res_lo_i =
				_mm256_cvtps_epi32(_mm256_add_ps(_mm256_div_ps(sum_v_lo, sum_w_lo), _mm256_set1_ps(0.5f)));
			__m256i v_res_hi_i =
				_mm256_cvtps_epi32(_mm256_add_ps(_mm256_div_ps(sum_v_hi, sum_w_hi), _mm256_set1_ps(0.5f)));

			// 【核心修复】跨 Lane 打包重排 (消除条纹)
			__m256i v_res_16 = _mm256_packus_epi32(v_res_lo_i, v_res_hi_i);
			v_res_16 = _mm256_permute4x64_epi64(v_res_16, 0xD8); // 控制字 0xD8 恢复顺序

			__m256i v_res_8 = _mm256_packus_epi16(v_res_16, v_res_16);
			v_res_8 = _mm256_permute4x64_epi64(v_res_8, 0xD8);

			_mm_storeu_si128((__m128i *)(dst_ptr + c), _mm256_castsi256_si128(v_res_8));
		}

		// 4. 边界标量处理 (必须同样使用 j*2)
		for (; c < cols - radius * 2; ++c) {
			float sw_sum = 1.0f, sv_sum = (float)src.ptr<uint8_t>(r)[c];
			uint8_t center = src.ptr<uint8_t>(r)[c];
			for (int i = -radius; i <= radius; ++i) {
				const uint8_t *s_row = src.ptr<uint8_t>(r + i * 2);
				for (int j = -radius; j <= radius; ++j) {
					if (i == 0 && j == 0)
						continue;
					uint8_t val = s_row[c + j * 2];
					float d2 = (i * i + j * j) * 4.0f;
					float r2 = (center - val) * (center - val);
					float w = std::exp(-(d2 * inv_2sigma_s2 + r2 / (2.0f * sigma_color * sigma_color)));
					sw_sum += w;
					sv_sum += w * val;
				}
			}
			dst_ptr[c] = (uint8_t)(sv_sum / sw_sum + 0.5f);
		}
	}
}

void lsc_simd_u16(cv::Mat &img, const cv::Mat &lsc_gain_map_int, float exposure) {
	int rows = img.rows;
	int cols = img.cols;

	int32_t exp_fix = static_cast<int32_t>(exposure * FIXED_SCALE);
	__m256i v_exp = _mm256_set1_epi32(exp_fix);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr_src = img.ptr<uint16_t>(r);
		const uint16_t *ptr_gain = lsc_gain_map_int.ptr<uint16_t>(r);
		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr_src + c));
			__m256i v_gain = _mm256_loadu_si256((const __m256i *)(ptr_gain + c));

			__m256i v_src_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
			__m256i v_gain_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));

			// [修改] 叠加曝光增益: Gain_New = (Gain_Map * Exp) >> FIXED_BITS
			v_gain_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_gain_lo, v_exp), FIXED_BITS);

			__m256i v_res_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), FIXED_BITS);

			__m256i v_src_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));
			__m256i v_gain_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));

			// [修改] 叠加曝光增益
			v_gain_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_gain_hi, v_exp), FIXED_BITS);

			__m256i v_res_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), FIXED_BITS);

			__m256i v_res = _mm256_packus_epi32(v_res_lo, v_res_hi);
			v_res = _mm256_permute4x64_epi64(v_res, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res);
		}
		// 处理剩余像素
		for (; c < cols; ++c) {
			// gain * exp
			uint32_t gain = (ptr_gain[c] * exp_fix) >> FIXED_BITS;
			uint32_t val = (uint32_t)ptr_src[c] * gain;
			val >>= FIXED_BITS;
			if (val > 65535)
				val = 65535;
			ptr_src[c] = (uint16_t)val;
		}
	}
}

void lsc_simd_u8(cv::Mat &img, const cv::Mat &lsc_gain_map_int, float exposure) {
	int rows = img.rows;
	int cols = img.cols;

	// [新增] 曝光增益
	int32_t exp_fix = static_cast<int32_t>(exposure * FIXED_SCALE);
	__m256i v_exp = _mm256_set1_epi32(exp_fix);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr_src = img.ptr<uint8_t>(r);
		const uint16_t *ptr_gain = lsc_gain_map_int.ptr<uint16_t>(r);
		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m128i v_src_small = _mm_loadu_si128((const __m128i *)(ptr_src + c));
			__m256i v_gain = _mm256_loadu_si256((const __m256i *)(ptr_gain + c));

			__m256i v_src_lo = _mm256_cvtepu8_epi32(v_src_small);
			__m256i v_gain_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));

			// [修改] 叠加曝光
			v_gain_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_gain_lo, v_exp), FIXED_BITS);

			__m256i v_res_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), FIXED_BITS);

			__m128i v_src_hi_small = _mm_unpackhi_epi64(v_src_small, v_src_small);
			__m256i v_src_hi = _mm256_cvtepu8_epi32(v_src_hi_small);
			__m256i v_gain_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));

			// [修改] 叠加曝光
			v_gain_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_gain_hi, v_exp), FIXED_BITS);

			__m256i v_res_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), FIXED_BITS);

			__m256i v_packed_16 = _mm256_packus_epi32(v_res_lo, v_res_hi);
			v_packed_16 = _mm256_permute4x64_epi64(v_packed_16, _MM_SHUFFLE(3, 1, 2, 0));

			__m128i v_packed_u8 =
				_mm_packus_epi16(_mm256_castsi256_si128(v_packed_16), _mm256_extracti128_si256(v_packed_16, 1));
			_mm_storeu_si128((__m128i *)(ptr_src + c), v_packed_u8);
		}
		for (; c < cols; ++c) {
			// gain * exp
			uint32_t gain = (ptr_gain[c] * exp_fix) >> FIXED_BITS;
			uint32_t val = (uint32_t)ptr_src[c] * gain;
			val >>= FIXED_BITS;
			if (val > 255)
				val = 255;
			ptr_src[c] = (uint8_t)val;
		}
	}
}

void awb_simd_u16(cv::Mat &img, const std::vector<float> &wbgains) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = FIXED_SCALE;
	uint16_t g_tl = static_cast<uint16_t>(wbgains[0] * scale);
	uint16_t g_tr = static_cast<uint16_t>(wbgains[1] * scale);
	uint16_t g_bl = static_cast<uint16_t>(wbgains[2] * scale);
	uint16_t g_br = static_cast<uint16_t>(wbgains[3] * scale);

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
			__m256i v0_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v0, 1));
			__m256i vg0_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain_row0));
			__m256i vg0_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain_row0, 1));
			v0_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v0_lo, vg0_lo), FIXED_BITS);
			v0_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v0_hi, vg0_hi), FIXED_BITS);
			v0 = _mm256_packus_epi32(v0_lo, v0_hi);
			v0 = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr0 + c), v0);

			__m256i v1 = _mm256_loadu_si256((const __m256i *)(ptr1 + c));
			__m256i v1_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v1));
			__m256i v1_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v1, 1));
			__m256i vg1_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain_row1));
			__m256i vg1_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain_row1, 1));
			v1_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v1_lo, vg1_lo), FIXED_BITS);
			v1_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v1_hi, vg1_hi), FIXED_BITS);
			v1 = _mm256_packus_epi32(v1_lo, v1_hi);
			v1 = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr1 + c), v1);
		}

		for (; c < cols; ++c) {
			uint32_t val0 = (uint32_t)ptr0[c] * ((c % 2) ? g_tr : g_tl);
			val0 >>= FIXED_BITS;
			ptr0[c] = (val0 > 65535) ? 65535 : (uint16_t)val0;
			uint32_t val1 = (uint32_t)ptr1[c] * ((c % 2) ? g_br : g_bl);
			val1 >>= FIXED_BITS;
			ptr1[c] = (val1 > 65535) ? 65535 : (uint16_t)val1;
		}
	}
}

void awb_simd_u8(cv::Mat &img, const std::vector<float> &wbgains) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = FIXED_SCALE;
	uint16_t g_tl = static_cast<uint16_t>(wbgains[0] * scale);
	uint16_t g_tr = static_cast<uint16_t>(wbgains[1] * scale);
	uint16_t g_bl = static_cast<uint16_t>(wbgains[2] * scale);
	uint16_t g_br = static_cast<uint16_t>(wbgains[3] * scale);

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

			__m256i v_src_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src_16));
			__m256i v_gain_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));
			__m256i v_src_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src_16, 1));
			__m256i v_gain_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));

			__m256i v_res_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), FIXED_BITS);
			__m256i v_res_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), FIXED_BITS);

			__m256i v_packed_16 = _mm256_packus_epi32(v_res_lo, v_res_hi);
			v_packed_16 = _mm256_permute4x64_epi64(v_packed_16, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i v_final =
				_mm_packus_epi16(_mm256_castsi256_si128(v_packed_16), _mm256_extracti128_si256(v_packed_16, 1));
			_mm_storeu_si128((__m128i *)(ptr_src + c), v_final);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint16_t gain = (c % 2 == 0) ? awb_0 : awb_1;
			uint32_t val = ((uint32_t)ptr_src[c] * gain) >> FIXED_BITS;
			if (val > 255)
				val = 255;
			ptr_src[c] = (uint8_t)val;
		}
	}
}

void lsc_awb_simd_u16(cv::Mat &img, float exposure, const cv::Mat &lsc_gain_map_int,
					  const std::vector<float> &wbgains) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = FIXED_SCALE;

	// [修改] 将 exposure 乘入 AWB 基础增益
	float exp_gain = exposure;

	uint16_t g_tl = static_cast<uint16_t>(wbgains[0] * scale * exp_gain);
	uint16_t g_tr = static_cast<uint16_t>(wbgains[1] * scale * exp_gain);
	uint16_t g_bl = static_cast<uint16_t>(wbgains[2] * scale * exp_gain);
	uint16_t g_br = static_cast<uint16_t>(wbgains[3] * scale * exp_gain);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr_src = img.ptr<uint16_t>(r);
		const uint16_t *ptr_lsc = lsc_gain_map_int.ptr<uint16_t>(r);
		__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

		int c = 0;
		for (; c <= cols - 16; c += 16) {
			__m256i v_lsc = _mm256_loadu_si256((const __m256i *)(ptr_lsc + c));

			__m256i v_lsc_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc));
			__m256i v_lsc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc, 1));
			__m256i v_awb_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
			__m256i v_awb_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

			// 这里不需要改，因为 awb 变量已经包含了 lscExp
			__m256i v_gain_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), FIXED_BITS);
			__m256i v_gain_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), FIXED_BITS);

			__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr_src + c));
			__m256i v_src_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
			__m256i v_src_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));

			v_src_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), FIXED_BITS);
			v_src_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), FIXED_BITS);

			__m256i v_res = _mm256_packus_epi32(v_src_lo, v_src_hi);
			v_res = _mm256_permute4x64_epi64(v_res, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint16_t lsc = ptr_lsc[c];
			uint16_t awb = (c % 2 == 0) ? awb_0 : awb_1;
			uint32_t total_gain = ((uint32_t)lsc * awb) >> FIXED_BITS;
			uint32_t val = ((uint32_t)ptr_src[c] * total_gain) >> FIXED_BITS;
			if (val > 65535)
				val = 65535;
			ptr_src[c] = static_cast<uint16_t>(val);
		}
	}
}

void lsc_awb_simd_u8(cv::Mat &img, float exposure, const cv::Mat &lsc_gain_map_int, const std::vector<float> &wbgains) {
	int rows = img.rows;
	int cols = img.cols;
	const float scale = FIXED_SCALE;

	// [修改] 引入 lscExp
	float exp_gain = exposure;

	uint16_t g_tl = static_cast<uint16_t>(wbgains[0] * scale * exp_gain);
	uint16_t g_tr = static_cast<uint16_t>(wbgains[1] * scale * exp_gain);
	uint16_t g_bl = static_cast<uint16_t>(wbgains[2] * scale * exp_gain);
	uint16_t g_br = static_cast<uint16_t>(wbgains[3] * scale * exp_gain);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr_src = img.ptr<uint8_t>(r);
		const uint16_t *ptr_lsc = lsc_gain_map_int.ptr<uint16_t>(r);
		__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

		int c = 0;
		for (; c <= cols - 32; c += 32) {
			__m256i v_src_32 = _mm256_loadu_si256((const __m256i *)(ptr_src + c));
			__m256i v_lsc_0 = _mm256_loadu_si256((const __m256i *)(ptr_lsc + c));
			__m256i v_lsc_1 = _mm256_loadu_si256((const __m256i *)(ptr_lsc + c + 16));

			auto process_half = [&](__m128i v_p_8, __m256i v_lsc_16) -> __m256i {
				__m256i v_p_lo = _mm256_cvtepu8_epi32(v_p_8);
				__m256i v_p_hi = _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(v_p_8, v_p_8));

				__m256i v_lsc_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc_16));
				__m256i v_lsc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc_16, 1));

				__m256i v_awb_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
				__m256i v_awb_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

				__m256i v_gain_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), FIXED_BITS);
				__m256i v_gain_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), FIXED_BITS);

				v_p_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_p_lo, v_gain_lo), FIXED_BITS);
				v_p_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_p_hi, v_gain_hi), FIXED_BITS);

				__m256i v_res_16 = _mm256_packus_epi32(v_p_lo, v_p_hi);
				return _mm256_permute4x64_epi64(v_res_16, _MM_SHUFFLE(3, 1, 2, 0));
			};

			__m256i v_res_0 = process_half(_mm256_castsi256_si128(v_src_32), v_lsc_0);
			__m256i v_res_1 = process_half(_mm256_extracti128_si256(v_src_32, 1), v_lsc_1);

			__m256i v_res_u8 = _mm256_packus_epi16(v_res_0, v_res_1);
			v_res_u8 = _mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));
			_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res_u8);
		}

		uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
		uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
		for (; c < cols; ++c) {
			uint16_t lsc = ptr_lsc[c];
			uint16_t awb = (c % 2 == 0) ? awb_0 : awb_1;
			uint32_t total_gain = ((uint32_t)lsc * awb) >> FIXED_BITS;
			uint32_t val = ((uint32_t)ptr_src[c] * total_gain) >> FIXED_BITS;
			if (val > 255)
				val = 255;
			ptr_src[c] = static_cast<uint8_t>(val);
		}
	}
}

void raw_to_8bit_with_gains_simd_u16(cv::Mat &src, cv::Mat &dst, const IspConfig &config, const cv::Mat &lsc_map) {
	int rows = src.rows;
	int cols = src.cols;

	uint16_t bl_val = static_cast<uint16_t>(config.black_level);
	__m256i v_bl = _mm256_set1_epi16(bl_val);

	float effective_range = static_cast<float>(config.white_level - config.black_level);
	if (effective_range < 1.0f)
		effective_range = 1.0f;

	// 计算总缩放因子 (使用 FIXED_SCALE)
	float total_scale_factor = (255.0f / effective_range) * config.lscExp * FIXED_SCALE;

	auto calc_gain = [&](float awb_g) -> uint16_t {
		float val = awb_g * total_scale_factor;
		if (val > 65535.0f)
			val = 65535.0f;
		return static_cast<uint16_t>(val);
	};

	uint16_t g_tl = calc_gain(config.awb_gains[0]);
	uint16_t g_tr = calc_gain(config.awb_gains[1]);
	uint16_t g_bl = calc_gain(config.awb_gains[2]);
	uint16_t g_br = calc_gain(config.awb_gains[3]);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

	bool has_lsc_runtime = !lsc_map.empty() && lsc_map.size() == src.size();

	// =========================================================
	// 核心计算 Kernel (定义在循环外，强制内联)
	// =========================================================
	// 处理 16个像素 (一个 __m256i)
	auto compute_block = [](const __m256i &v_src, const __m256i &v_lsc, const __m256i &v_awb) -> __m256i {
		// 1. Unpack source to 32-bit (Pixel - BL)
		__m256i v_src_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
		__m256i v_src_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));

		// 2. Unpack LSC & AWB to 32-bit
		__m256i v_lsc_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc));
		__m256i v_lsc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc, 1));

		__m256i v_awb_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
		__m256i v_awb_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

		// 3. Combine Gains: (LSC * AWB) >> Shift
		// 提前合并增益，减少一次对 v_src 的乘法
		__m256i v_gain_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), FIXED_BITS);
		__m256i v_gain_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), FIXED_BITS);

		// 4. Apply Gain: (Pixel * Gain) >> Shift
		v_src_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_gain_lo), FIXED_BITS);
		v_src_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_gain_hi), FIXED_BITS);

		// 5. Pack 32-bit back to 16-bit (Satruated)
		return _mm256_packus_epi32(v_src_lo, v_src_hi);
		// 注意：packus_epi32 后的数据顺序在 256位下需要 permute，
		// 但我们下面马上要拆成
		// 128位处理，所以这里暂时保持乱序即可，或者统一在最后处理
	};

	// 无 LSC 版本的 Kernel
	auto compute_block_no_lsc = [](const __m256i &v_src, const __m256i &v_awb) -> __m256i {
		__m256i v_src_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
		__m256i v_src_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));

		__m256i v_awb_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
		__m256i v_awb_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

		// Apply AWB Gain Only
		v_src_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_lo, v_awb_lo), FIXED_BITS);
		v_src_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_src_hi, v_awb_hi), FIXED_BITS);

		return _mm256_packus_epi32(v_src_lo, v_src_hi);
	};

	// =========================================================
	// 模板化 Loop
	// =========================================================
	auto run_loop = [&](auto has_lsc_tag) {
		constexpr bool HAS_LSC = has_lsc_tag.value;

#pragma omp parallel for
		for (int r = 0; r < rows; ++r) {
			uint8_t *pSrc = src.ptr<uint8_t>(r);
			uint8_t *pDst = dst.ptr<uint8_t>(r);

			const uint16_t *lsc_ptr = HAS_LSC ? lsc_map.ptr<uint16_t>(r) : nullptr;
			__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

			int c = 0;
			for (; c <= cols - 16; c += 16) {
				// Load 16 pixels (16-bit)
				__m256i v_src = _mm256_loadu_si256((const __m256i *)(pSrc + c));
				v_src = _mm256_subs_epu16(v_src, v_bl); // Subtract BL

				__m256i v_res_16;

				if constexpr (HAS_LSC) {
					__m256i v_lsc = _mm256_loadu_si256((const __m256i *)(lsc_ptr + c));
					v_res_16 = compute_block(v_src, v_lsc, v_awb);
				} else {
					v_res_16 = compute_block_no_lsc(v_src, v_awb);
				}

				// 此时 v_res_16 是 256bit (16个 u16)，顺序是乱的 (因为
				// packus_epi32) [A0..A3 B0..B3 A4..A7 B4..B7] (32-bit blocks)
				// 我们需要 permute 恢复顺序
				v_res_16 = _mm256_permute4x64_epi64(v_res_16, _MM_SHUFFLE(3, 1, 2, 0));

				// 压缩 16-bit -> 8-bit
				// 由于 packus_epi16 需要两个 128-bit 输入，我们将 256-bit 拆开
				__m128i v_res_lo_128 = _mm256_castsi256_si128(v_res_16);
				__m128i v_res_hi_128 = _mm256_extracti128_si256(v_res_16, 1);

				// Pack 两个 128-bit (16x u16) -> 一个 128-bit (16x u8)
				__m128i v_res_u8 = _mm_packus_epi16(v_res_lo_128, v_res_hi_128);

				_mm_storeu_si128((__m128i *)(pDst + c), v_res_u8);
			}

			// Scalar Cleanup
			uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
			uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;

			for (; c < cols; ++c) {
				uint32_t val = pSrc[c];
				val = (val > bl_val) ? (val - bl_val) : 0;

				if constexpr (HAS_LSC) {
					val = (val * lsc_ptr[c]) >> FIXED_BITS;
				}

				uint32_t awb = (c % 2 == 0) ? awb_0 : awb_1;
				val = (val * awb) >> FIXED_BITS;
				if (val > 255)
					val = 255;
				pDst[c] = static_cast<uint8_t>(val);
			}
		}
	};

	if (has_lsc_runtime) {
		run_loop(std::true_type{});
	} else {
		run_loop(std::false_type{});
	}
}

void raw_to_8bit_with_gains_simd_u8(cv::Mat &src, cv::Mat &dst, const IspConfig &config, const cv::Mat &lsc_map) {
	int rows = src.rows;
	int cols = src.cols;

	int bl_shift = (config.bitDepth > 8) ? (config.bitDepth - 8) : 0;
	uint8_t bl_val = static_cast<uint8_t>(config.black_level >> bl_shift);
	__m256i v_bl = _mm256_set1_epi8(bl_val);

	float effective_range = 255.0f - bl_val;
	if (effective_range < 1.0f)
		effective_range = 1.0f;

	// 计算缩放因子
	float total_scale_factor = (255.0f / effective_range) * config.lscExp * FIXED_SCALE;

	auto calc_gain = [&](float awb_g) -> uint16_t {
		float val = awb_g * total_scale_factor;
		if (val > 65535.0f)
			val = 65535.0f;
		return static_cast<uint16_t>(val);
	};

	uint16_t g_tl = calc_gain(config.awb_gains[0]);
	uint16_t g_tr = calc_gain(config.awb_gains[1]);
	uint16_t g_bl = calc_gain(config.awb_gains[2]);
	uint16_t g_br = calc_gain(config.awb_gains[3]);

	uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
	__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
	uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
	__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

	bool has_lsc_runtime = !lsc_map.empty() && lsc_map.size() == src.size();

	// 定义核心处理逻辑的宏或内联 lambda (避免捕获开销，纯计算)
	// 这里使用 Lambda static 技巧，强制内联
	auto compute_block = [](const __m256i &v_src_part, const __m256i &v_lsc_part,
							const __m256i &v_awb_part) -> __m256i {
		// Unpack 8-bit -> 32-bit (Part 1)
		__m256i v_p_lo = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(v_src_part));
		// Unpack 8-bit -> 32-bit (Part 2)
		__m256i v_p_hi = _mm256_cvtepu8_epi32(_mm256_extracti128_si256(v_src_part, 1));

		// Unpack LSC & AWB to 32-bit
		__m256i v_lsc_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc_part));
		__m256i v_lsc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc_part, 1));

		__m256i v_awb_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb_part));
		__m256i v_awb_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb_part, 1));

		// Combine Gains: (LSC * AWB) >> Shift
		__m256i v_gain_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), FIXED_BITS);
		__m256i v_gain_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), FIXED_BITS);

		// Apply Gain: (Pixel * Gain) >> Shift
		v_p_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_p_lo, v_gain_lo), FIXED_BITS);
		v_p_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_p_hi, v_gain_hi), FIXED_BITS);

		// Pack back to 16-bit
		return _mm256_packus_epi32(v_p_lo, v_p_hi);
	};

	// 无 LSC 版本的 Compute (减少运算)
	auto compute_block_no_lsc = [](const __m256i &v_src_part, const __m256i &v_awb_part) -> __m256i {
		__m256i v_p_lo = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(v_src_part));
		__m256i v_p_hi = _mm256_cvtepu8_epi32(_mm256_extracti128_si256(v_src_part, 1));

		__m256i v_awb_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb_part));
		__m256i v_awb_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb_part, 1));

		// Apply AWB Gain Only
		v_p_lo = _mm256_srli_epi32(_mm256_mullo_epi32(v_p_lo, v_awb_lo), FIXED_BITS);
		v_p_hi = _mm256_srli_epi32(_mm256_mullo_epi32(v_p_hi, v_awb_hi), FIXED_BITS);

		return _mm256_packus_epi32(v_p_lo, v_p_hi);
	};

	// =========================================================
	// 模板化 Loop，消除重复代码
	// =========================================================
	auto run_loop = [&](auto has_lsc_tag) {
		constexpr bool HAS_LSC = has_lsc_tag.value;

#pragma omp parallel for
		for (int r = 0; r < rows; ++r) {
			uint8_t *pSrc = src.ptr<uint8_t>(r);
			uint8_t *pDst = dst.ptr<uint8_t>(r);

			// 优化：指针定义
			const uint16_t *lsc_ptr = HAS_LSC ? lsc_map.ptr<uint16_t>(r) : nullptr;
			__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

			int c = 0;
			for (; c <= cols - 32; c += 32) {
				// Load 32 pixels
				__m256i v_src_32 = _mm256_loadu_si256((const __m256i *)(pSrc + c));
				v_src_32 = _mm256_subs_epu8(v_src_32, v_bl); // Subtract BL

				// Split into two 128-bit lanes (16 pixels each) expanded to
				// 256-bit Lane 0 (Pixels 0-15)
				__m256i v_src_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_src_32));
				// Lane 1 (Pixels 16-31)
				__m256i v_src_1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v_src_32, 1));

				// Result holders
				__m256i v_res_0_16, v_res_1_16;

				if constexpr (HAS_LSC) {
					__m256i v_lsc_0 = _mm256_loadu_si256((const __m256i *)(lsc_ptr + c));
					__m256i v_lsc_1 = _mm256_loadu_si256((const __m256i *)(lsc_ptr + c + 16));

					v_res_0_16 = compute_block(v_src_0, v_lsc_0, v_awb);
					v_res_1_16 = compute_block(v_src_1, v_lsc_1, v_awb);
				} else {
					v_res_0_16 = compute_block_no_lsc(v_src_0, v_awb);
					v_res_1_16 = compute_block_no_lsc(v_src_1, v_awb);
				}

				// Pack 16-bit results back to 8-bit
				__m256i v_res_u8 = _mm256_packus_epi16(v_res_0_16, v_res_1_16);
				// Permute to fix order after AVX2 pack
				v_res_u8 = _mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));

				_mm256_storeu_si256((__m256i *)(pDst + c), v_res_u8);
			}

			// Scalar Cleanup
			uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
			uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;

			for (; c < cols; ++c) {
				uint32_t val = pSrc[c];
				val = (val > bl_val) ? (val - bl_val) : 0;

				if constexpr (HAS_LSC) {
					val = (val * lsc_ptr[c]) >> FIXED_BITS;
				}

				uint32_t awb = (c % 2 == 0) ? awb_0 : awb_1;
				val = (val * awb) >> FIXED_BITS;
				if (val > 255)
					val = 255;
				pDst[c] = static_cast<uint8_t>(val);
			}
		}
	};

	if (has_lsc_runtime) {
		run_loop(std::true_type{});
	} else {
		run_loop(std::false_type{});
	}
}

void ccm_fixed_u16(cv::Mat &img, const std::vector<int32_t> &ccm_matrix_int) {
	int rows = img.rows;
	int cols = img.cols;

	// [修正 1] 显式定义标量系数，供 SIMD 初始化和底部的标量循环使用
	const int32_t c00 = ccm_matrix_int[0], c01 = ccm_matrix_int[1], c02 = ccm_matrix_int[2];
	const int32_t c10 = ccm_matrix_int[3], c11 = ccm_matrix_int[4], c12 = ccm_matrix_int[5];
	const int32_t c20 = ccm_matrix_int[6], c21 = ccm_matrix_int[7], c22 = ccm_matrix_int[8];

	// 1. 准备矩阵系数 (广播到 SIMD 寄存器)
	const __m256i v_c00 = _mm256_set1_epi32(c00);
	const __m256i v_c01 = _mm256_set1_epi32(c01);
	const __m256i v_c02 = _mm256_set1_epi32(c02);
	const __m256i v_c10 = _mm256_set1_epi32(c10);
	const __m256i v_c11 = _mm256_set1_epi32(c11);
	const __m256i v_c12 = _mm256_set1_epi32(c12);
	const __m256i v_c20 = _mm256_set1_epi32(c20);
	const __m256i v_c21 = _mm256_set1_epi32(c21);
	const __m256i v_c22 = _mm256_set1_epi32(c22);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint16_t *ptr = img.ptr<uint16_t>(r);
		int c = 0;

		// SIMD 循环：每次处理 8 个像素
		for (; c <= cols - 8; c += 8) {
			// 手动 Load (模拟 Gather)
			__m256i v_b = _mm256_setr_epi32(ptr[c * 3 + 0], ptr[c * 3 + 3], ptr[c * 3 + 6], ptr[c * 3 + 9],
											ptr[c * 3 + 12], ptr[c * 3 + 15], ptr[c * 3 + 18], ptr[c * 3 + 21]);

			__m256i v_g = _mm256_setr_epi32(ptr[c * 3 + 1], ptr[c * 3 + 4], ptr[c * 3 + 7], ptr[c * 3 + 10],
											ptr[c * 3 + 13], ptr[c * 3 + 16], ptr[c * 3 + 19], ptr[c * 3 + 22]);

			__m256i v_r = _mm256_setr_epi32(ptr[c * 3 + 2], ptr[c * 3 + 5], ptr[c * 3 + 8], ptr[c * 3 + 11],
											ptr[c * 3 + 14], ptr[c * 3 + 17], ptr[c * 3 + 20], ptr[c * 3 + 23]);

			// --- 矩阵乘法 ---
			// R_new
			__m256i v_r_new = _mm256_mullo_epi32(v_r, v_c00);
			v_r_new = _mm256_add_epi32(v_r_new, _mm256_mullo_epi32(v_g, v_c01));
			v_r_new = _mm256_add_epi32(v_r_new, _mm256_mullo_epi32(v_b, v_c02));
			v_r_new = _mm256_srai_epi32(v_r_new, FIXED_BITS); // Shift

			// G_new
			__m256i v_g_new = _mm256_mullo_epi32(v_r, v_c10);
			v_g_new = _mm256_add_epi32(v_g_new, _mm256_mullo_epi32(v_g, v_c11));
			v_g_new = _mm256_add_epi32(v_g_new, _mm256_mullo_epi32(v_b, v_c12));
			v_g_new = _mm256_srai_epi32(v_g_new, FIXED_BITS);

			// B_new
			__m256i v_b_new = _mm256_mullo_epi32(v_r, v_c20);
			v_b_new = _mm256_add_epi32(v_b_new, _mm256_mullo_epi32(v_g, v_c21));
			v_b_new = _mm256_add_epi32(v_b_new, _mm256_mullo_epi32(v_b, v_c22));
			v_b_new = _mm256_srai_epi32(v_b_new, FIXED_BITS);

			// --- Pack & Store ---
			__m256i v_bg = _mm256_packus_epi32(v_b_new, v_g_new);
			__m256i v_r0 = _mm256_packus_epi32(v_r_new, _mm256_setzero_si256());

			uint16_t *p = ptr + c * 3;
			// Lane 0
			p[0] = _mm256_extract_epi16(v_bg, 0);
			p[1] = _mm256_extract_epi16(v_bg, 4);
			p[2] = _mm256_extract_epi16(v_r0, 0);
			p[3] = _mm256_extract_epi16(v_bg, 1);
			p[4] = _mm256_extract_epi16(v_bg, 5);
			p[5] = _mm256_extract_epi16(v_r0, 1);
			p[6] = _mm256_extract_epi16(v_bg, 2);
			p[7] = _mm256_extract_epi16(v_bg, 6);
			p[8] = _mm256_extract_epi16(v_r0, 2);
			p[9] = _mm256_extract_epi16(v_bg, 3);
			p[10] = _mm256_extract_epi16(v_bg, 7);
			p[11] = _mm256_extract_epi16(v_r0, 3);

			// Lane 1
			p[12] = _mm256_extract_epi16(v_bg, 8);
			p[13] = _mm256_extract_epi16(v_bg, 12);
			p[14] = _mm256_extract_epi16(v_r0, 8);
			p[15] = _mm256_extract_epi16(v_bg, 9);
			p[16] = _mm256_extract_epi16(v_bg, 13);
			p[17] = _mm256_extract_epi16(v_r0, 9);
			p[18] = _mm256_extract_epi16(v_bg, 10);
			p[19] = _mm256_extract_epi16(v_bg, 14);
			p[20] = _mm256_extract_epi16(v_r0, 10);
			p[21] = _mm256_extract_epi16(v_bg, 11);
			p[22] = _mm256_extract_epi16(v_bg, 15);
			p[23] = _mm256_extract_epi16(v_r0, 11);
		}

		// 剩余像素处理 (现在 c00...c22 已经定义了，不会报错)
		for (; c < cols; ++c) {
			int idx = c * 3;
			int32_t b_val = ptr[idx + 0];
			int32_t g_val = ptr[idx + 1];
			int32_t r_val = ptr[idx + 2];

			int32_t r_new = (r_val * c00 + g_val * c01 + b_val * c02) >> FIXED_BITS;
			int32_t g_new = (r_val * c10 + g_val * c11 + b_val * c12) >> FIXED_BITS;
			int32_t b_new = (r_val * c20 + g_val * c21 + b_val * c22) >> FIXED_BITS;

			ptr[idx + 0] = cv::saturate_cast<uint16_t>(std::max(0, b_new));
			ptr[idx + 1] = cv::saturate_cast<uint16_t>(std::max(0, g_new));
			ptr[idx + 2] = cv::saturate_cast<uint16_t>(std::max(0, r_new));
		}
	}
}

void ccm_fixed_u8(cv::Mat &img, const std::vector<int32_t> &ccm_matrix_int) {
	int rows = img.rows;
	int cols = img.cols;

	// [修正 1] 同样显式定义标量系数
	const int32_t c00 = ccm_matrix_int[0], c01 = ccm_matrix_int[1], c02 = ccm_matrix_int[2];
	const int32_t c10 = ccm_matrix_int[3], c11 = ccm_matrix_int[4], c12 = ccm_matrix_int[5];
	const int32_t c20 = ccm_matrix_int[6], c21 = ccm_matrix_int[7], c22 = ccm_matrix_int[8];

	const __m256i v_c00 = _mm256_set1_epi32(c00);
	const __m256i v_c01 = _mm256_set1_epi32(c01);
	const __m256i v_c02 = _mm256_set1_epi32(c02);
	const __m256i v_c10 = _mm256_set1_epi32(c10);
	const __m256i v_c11 = _mm256_set1_epi32(c11);
	const __m256i v_c12 = _mm256_set1_epi32(c12);
	const __m256i v_c20 = _mm256_set1_epi32(c20);
	const __m256i v_c21 = _mm256_set1_epi32(c21);
	const __m256i v_c22 = _mm256_set1_epi32(c22);

	// 8位 gather 索引 (Stride=3)
	__m256i v_idx_base = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
	__m256i v_idx_g = _mm256_add_epi32(v_idx_base, _mm256_set1_epi32(1));
	__m256i v_idx_r = _mm256_add_epi32(v_idx_base, _mm256_set1_epi32(2));
	__m256i v_mask = _mm256_set1_epi32(0xFF);

#pragma omp parallel for
	for (int r = 0; r < rows; ++r) {
		uint8_t *ptr = img.ptr<uint8_t>(r);
		int c = 0;

		// SIMD 循环：8 像素
		for (; c <= cols - 8; c += 8) {
			// Gather
			__m256i v_b = _mm256_i32gather_epi32((const int *)(ptr + c * 3), v_idx_base, 1);
			__m256i v_g = _mm256_i32gather_epi32((const int *)(ptr + c * 3), v_idx_g, 1);
			__m256i v_r = _mm256_i32gather_epi32((const int *)(ptr + c * 3), v_idx_r, 1);

			v_b = _mm256_and_si256(v_b, v_mask);
			v_g = _mm256_and_si256(v_g, v_mask);
			v_r = _mm256_and_si256(v_r, v_mask);

			// --- 矩阵乘法 ---
			__m256i v_r_new = _mm256_mullo_epi32(v_r, v_c00);
			v_r_new = _mm256_add_epi32(v_r_new, _mm256_mullo_epi32(v_g, v_c01));
			v_r_new = _mm256_add_epi32(v_r_new, _mm256_mullo_epi32(v_b, v_c02));
			v_r_new = _mm256_srai_epi32(v_r_new, FIXED_BITS);

			__m256i v_g_new = _mm256_mullo_epi32(v_r, v_c10);
			v_g_new = _mm256_add_epi32(v_g_new, _mm256_mullo_epi32(v_g, v_c11));
			v_g_new = _mm256_add_epi32(v_g_new, _mm256_mullo_epi32(v_b, v_c12));
			v_g_new = _mm256_srai_epi32(v_g_new, FIXED_BITS);

			__m256i v_b_new = _mm256_mullo_epi32(v_r, v_c20);
			v_b_new = _mm256_add_epi32(v_b_new, _mm256_mullo_epi32(v_g, v_c21));
			v_b_new = _mm256_add_epi32(v_b_new, _mm256_mullo_epi32(v_b, v_c22));
			v_b_new = _mm256_srai_epi32(v_b_new, FIXED_BITS);

			// [修正 2] 避免使用 extract lambda，改用 Store 到临时数组
			// 这种方法避免了 "must be constant integer"
			// 错误，且编译器优化后性能极高
			int32_t buf_r[8], buf_g[8], buf_b[8];
			_mm256_storeu_si256((__m256i *)buf_r, v_r_new);
			_mm256_storeu_si256((__m256i *)buf_g, v_g_new);
			_mm256_storeu_si256((__m256i *)buf_b, v_b_new);

			// 循环写入 (手动饱和)
			uint8_t *p_dst = ptr + c * 3;
			for (int k = 0; k < 8; ++k) {
				p_dst[k * 3 + 0] = cv::saturate_cast<uint8_t>(buf_b[k]);
				p_dst[k * 3 + 1] = cv::saturate_cast<uint8_t>(buf_g[k]);
				p_dst[k * 3 + 2] = cv::saturate_cast<uint8_t>(buf_r[k]);
			}
		}

		// 剩余像素处理
		for (; c < cols; ++c) {
			int idx = c * 3;
			int32_t b_val = ptr[idx + 0];
			int32_t g_val = ptr[idx + 1];
			int32_t r_val = ptr[idx + 2];

			int32_t r_new = (r_val * c00 + g_val * c01 + b_val * c02) >> FIXED_BITS;
			int32_t g_new = (r_val * c10 + g_val * c11 + b_val * c12) >> FIXED_BITS;
			int32_t b_new = (r_val * c20 + g_val * c21 + b_val * c22) >> FIXED_BITS;

			ptr[idx + 0] = cv::saturate_cast<uint8_t>(std::max(0, b_new));
			ptr[idx + 1] = cv::saturate_cast<uint8_t>(std::max(0, g_new));
			ptr[idx + 2] = cv::saturate_cast<uint8_t>(std::max(0, r_new));
		}
	}
}

inline void row_simd_u16_to_u8(const uint16_t *src, uint8_t *dst, int width, float alpha, float beta) {
	int x = 0;

	__m256 v_alpha = _mm256_set1_ps(alpha);
	__m256 v_beta = _mm256_set1_ps(beta);

	for (; x <= width - 16; x += 16) {
		__m256i v_src_u16 = _mm256_loadu_si256((const __m256i *)(src + x));

		__m256i v_src_i32_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src_u16));
		__m256i v_src_i32_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src_u16, 1));

		__m256 v_f32_lo = _mm256_cvtepi32_ps(v_src_i32_lo);
		__m256 v_f32_hi = _mm256_cvtepi32_ps(v_src_i32_hi);

		v_f32_lo = _mm256_fmadd_ps(v_f32_lo, v_alpha, v_beta);
		v_f32_hi = _mm256_fmadd_ps(v_f32_hi, v_alpha, v_beta);

		__m256i v_res_i32_lo = _mm256_cvtps_epi32(v_f32_lo);
		__m256i v_res_i32_hi = _mm256_cvtps_epi32(v_f32_hi);

		__m256i v_res_i16 = _mm256_packus_epi32(v_res_i32_lo, v_res_i32_hi);

		__m256i v_res_u8_scrambled = _mm256_packus_epi16(v_res_i16, _mm256_setzero_si256());

		v_res_i16 = _mm256_permute4x64_epi64(v_res_i16, _MM_SHUFFLE(3, 1, 2, 0));

		__m256i v_zero = _mm256_setzero_si256();
		__m256i v_res_u8 = _mm256_packus_epi16(v_res_i16, v_zero);

		v_res_u8 = _mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));

		_mm_storeu_si128((__m128i *)(dst + x), _mm256_castsi256_si128(v_res_u8));
	}

	for (; x < width; ++x) {
		float val = (float)src[x];
		float res = val * alpha + beta;
		res = (res > 255.0f) ? 255.0f : (res < 0.0f ? 0.0f : res);
		dst[x] = (uint8_t)res;
	}
}

inline void row_simd_u8_to_u8(const uint8_t *src, uint8_t *dst, int width, float alpha, float beta) {
	int x = 0;
	__m256 v_alpha = _mm256_set1_ps(alpha);
	__m256 v_beta = _mm256_set1_ps(beta);

	for (; x <= width - 16; x += 16) {
		__m128i v_src_u8 = _mm_loadu_si128((const __m128i *)(src + x));
		__m256i v_src_u16 = _mm256_cvtepu8_epi16(v_src_u8);

		__m256i v_src_i32_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src_u16));
		__m256i v_src_i32_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src_u16, 1));

		__m256 v_f32_lo = _mm256_cvtepi32_ps(v_src_i32_lo);
		__m256 v_f32_hi = _mm256_cvtepi32_ps(v_src_i32_hi);

		v_f32_lo = _mm256_fmadd_ps(v_f32_lo, v_alpha, v_beta);
		v_f32_hi = _mm256_fmadd_ps(v_f32_hi, v_alpha, v_beta);

		__m256i v_res_i32_lo = _mm256_cvtps_epi32(v_f32_lo);
		__m256i v_res_i32_hi = _mm256_cvtps_epi32(v_f32_hi);

		__m256i v_res_i16 = _mm256_packus_epi32(v_res_i32_lo, v_res_i32_hi);
		v_res_i16 = _mm256_permute4x64_epi64(v_res_i16, _MM_SHUFFLE(3, 1, 2, 0));

		__m256i v_res_u8 = _mm256_packus_epi16(v_res_i16, _mm256_setzero_si256());
		v_res_u8 = _mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));

		_mm_storeu_si128((__m128i *)(dst + x), _mm256_castsi256_si128(v_res_u8));
	}

	for (; x < width; ++x) {
		float val = (float)src[x];
		float res = val * alpha + beta;
		res = (res > 255.0f) ? 255.0f : (res < 0.0f ? 0.0f : res);
		dst[x] = (uint8_t)res;
	}
}

} // namespace

// ============================================================================
// LFIsp 类实现
// ============================================================================

LFIsp::LFIsp() { cv::setNumThreads(cv::getNumberOfCPUs()); }

LFIsp::LFIsp(const cv::Mat &lfp_img) {
	cv::setNumThreads(cv::getNumberOfCPUs());
	set_lf_img(lfp_img);
}

LFIsp::LFIsp(const cv::Mat &lfp_img, const cv::Mat &wht_img, const IspConfig &config) {
	cv::setNumThreads(cv::getNumberOfCPUs());
	set_lf_img(lfp_img);
	initConfig(wht_img, config);
}

LFIsp &LFIsp::print_config(const IspConfig &config) {
	std::cout << "\n================ [LFIsp Config] ================" << std::endl;
	std::cout << "Bayer Pattern : " << bayerToString(config.bayer) << std::endl;
	std::cout << "Bit Depth     : " << config.bitDepth << "-bit" << std::endl;
	std::cout << "Black Level   : " << config.black_level << std::endl;
	std::cout << "White Level   : " << config.white_level << std::endl;

	std::cout << "AWB Gains     : [ ";
	std::cout << std::fixed << std::setprecision(3);
	if (config.awb_gains.size() >= 4) {
		std::cout << "Gr:" << config.awb_gains[0] << ", ";
		std::cout << "R :" << config.awb_gains[1] << ", ";
		std::cout << "B :" << config.awb_gains[2] << ", ";
		std::cout << "Gb:" << config.awb_gains[3];
	} else {
		for (float g : config.awb_gains) std::cout << g << " ";
	}
	std::cout << " ]" << std::endl;

	std::cout << "Gamma         : " << config.gamma << std::endl;

	std::cout << "CCM Matrix    :" << std::endl;
	if (config.ccm_matrix.size() == 9) {
		for (int i = 0; i < 3; ++i) {
			std::cout << "                [ ";
			for (int j = 0; j < 3; ++j) {
				float val = config.ccm_matrix[i * 3 + j];
				std::cout << std::showpos << std::setw(8) << val << " ";
			}
			std::cout << std::noshowpos << "]" << std::endl;
		}
	} else {
		std::cout << "                (Invalid size: " << config.ccm_matrix.size() << ")" << std::endl;
	}
	std::cout << "================================================" << std::endl;
	std::cout << std::defaultfloat;
	return *this;
}

void LFIsp::parseJsonToConfig(const json &j, IspConfig &config) {
	if (j.contains("bay")) {
		std::string bay_str = j["bay"].get<std::string>();
		if (bay_str == "GRBG")
			config.bayer = BayerPattern::GRBG;
		else if (bay_str == "RGGB")
			config.bayer = BayerPattern::RGGB;
		else if (bay_str == "GBRG")
			config.bayer = BayerPattern::GBRG;
		else if (bay_str == "BGGR")
			config.bayer = BayerPattern::BGGR;
	}

	if (j.contains("bit")) {
		config.bitDepth = j["bit"].get<int>();
	}

	if (j.contains("blc")) {
		const auto &blc = j["blc"];
		if (blc.contains("black") && blc["black"].is_array() && !blc["black"].empty()) {
			config.black_level = blc["black"][0].get<int>();
		}
		if (blc.contains("white") && blc["white"].is_array() && !blc["white"].empty()) {
			config.white_level = blc["white"][0].get<int>();
		}
	}

	if (j.contains("awb") && j["awb"].is_array()) {
		config.awb_gains = j["awb"].get<std::vector<float>>();
	}

	if (j.contains("ccm") && j["ccm"].is_array()) {
		config.ccm_matrix = j["ccm"].get<std::vector<float>>();
	}

	if (j.contains("gam")) {
		config.gamma = j["gam"].get<float>();
	}
} // parseJsonToConfig

LFIsp &LFIsp::set_lf_img(const cv::Mat &img) {
	if (img.empty())
		throw std::runtime_error("LF image is empty.");
	lfp_img_ = img;
	return *this;
} // set_lf_img

LFIsp &LFIsp::initConfig(const cv::Mat &img, const IspConfig &config) {
	if (img.empty())
		throw std::runtime_error("White image is empty.");
	prepare_lsc_maps(img, config.black_level);
	prepare_ccm_fixed_point(config.ccm_matrix);
	prepare_gamma_lut(config.gamma, 8);

	return *this;
} // initConfig

std::string LFIsp::bayerToString(BayerPattern p) {
	switch (p) {
		case BayerPattern::GRBG:
			return "GRBG";
		case BayerPattern::RGGB:
			return "RGGB";
		case BayerPattern::GBRG:
			return "GBRG";
		case BayerPattern::BGGR:
			return "BGGR";
		case BayerPattern::NONE:
			return "NONE";
		default:
			return "Unknown";
	}
} // bayerToString

// ============================================================================
// 标量处理流程 (Scalar Implementation)
// ============================================================================

LFIsp &LFIsp::blc(int black_level, int white_level) {
	if (lfp_img_.empty()) {
		return *this;
	}
	double bl = static_cast<double>(black_level);
	double wl = static_cast<double>(white_level);

	double effective_range = wl - bl;
	if (effective_range < 1.0)
		effective_range = 1.0; // 防止除零

	double alpha = 255.0 / effective_range;
	double beta = -bl * alpha;

	lfp_img_.convertTo(lfp_img_, CV_8U, alpha, beta);
	return *this;
} // blc

LFIsp &LFIsp::dpc(int threshold) {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.depth() != CV_8U) {
		dpc_scalar_impl<uint16_t>(lfp_img_, threshold);
	} else {
		dpc_scalar_impl<uint8_t>(lfp_img_, threshold);
	}
	return *this;
} // dpc

LFIsp &LFIsp::lsc(float exposure) {
	if (lfp_img_.empty() || lsc_gain_map_.empty())
		return *this;
	if (lfp_img_.size() != lsc_gain_map_.size())
		return *this;

	cv::Mat float_img;
	lfp_img_.convertTo(float_img, CV_32F);
	cv::multiply(float_img, lsc_gain_map_, float_img);
	float_img.convertTo(lfp_img_, lfp_img_.type());
	return *this;
} // lsc

LFIsp &LFIsp::rawnr(float sigma_spatial, float sigma_color) {
	if (lfp_img_.empty() || lfp_img_.type() != CV_8UC1)
		return *this;

	// 1. 初始化 4 个子通道 (R, Gr, Gb, B)
	// 尺寸为原图的一半
	int sub_rows = lfp_img_.rows / 2;
	int sub_cols = lfp_img_.cols / 2;
	std::vector<cv::Mat> channels(4);
	for (int i = 0; i < 4; ++i) {
		channels[i] = cv::Mat(sub_rows, sub_cols, CV_8UC1);
	}

	// 2. 像素分发 (De-interleaving)
	// 按照 2x2 矩阵跨步提取，对应 Bayer 格式的四个位置
	for (int r = 0; r < sub_rows; ++r) {
		const uchar *p0 = lfp_img_.ptr<uchar>(2 * r);	  // 偶数行
		const uchar *p1 = lfp_img_.ptr<uchar>(2 * r + 1); // 奇数行

		uchar *c0 = channels[0].ptr<uchar>(r);
		uchar *c1 = channels[1].ptr<uchar>(r);
		uchar *c2 = channels[2].ptr<uchar>(r);
		uchar *c3 = channels[3].ptr<uchar>(r);

		for (int c = 0; c < sub_cols; ++c) {
			c0[c] = p0[2 * c];	   // 位置 (0,0) - 如 Gr
			c1[c] = p0[2 * c + 1]; // 位置 (0,1) - 如 R
			c2[c] = p1[2 * c];	   // 位置 (1,0) - 如 B
			c3[c] = p1[2 * c + 1]; // 位置 (1,1) - 如 Gb
		}
	}

	//

	// 3. 对 4 个通道分别应用双边滤波
	// 直接在 CV_8U 上操作，符合 baseline 的直接性要求
	for (int i = 0; i < 4; ++i) {
		cv::Mat filtered;
		// d=0 表示滤波器直径由 sigma_spatial 自动导出
		cv::bilateralFilter(channels[i], filtered, 0, sigma_color, sigma_spatial);
		channels[i] = filtered;
	}

	// 4. 将降噪后的通道重新交织回原图 (Re-interleaving)
	for (int r = 0; r < sub_rows; ++r) {
		uchar *p0 = lfp_img_.ptr<uchar>(2 * r);
		uchar *p1 = lfp_img_.ptr<uchar>(2 * r + 1);

		uchar *c0 = channels[0].ptr<uchar>(r);
		uchar *c1 = channels[1].ptr<uchar>(r);
		uchar *c2 = channels[2].ptr<uchar>(r);
		uchar *c3 = channels[3].ptr<uchar>(r);

		for (int c = 0; c < sub_cols; ++c) {
			p0[2 * c] = c0[c];
			p0[2 * c + 1] = c1[c];
			p1[2 * c] = c2[c];
			p1[2 * c + 1] = c3[c];
		}
	}

	return *this;
} // rawnr

LFIsp &LFIsp::awb(const std::vector<float> &wbgains) {
	if (lfp_img_.empty())
		return *this;

	if (wbgains.size() != 4)
		return *this;

	float g_tl = wbgains[0];
	float g_tr = wbgains[1];
	float g_bl = wbgains[2];
	float g_br = wbgains[3];

	int rows = lfp_img_.rows;
	int cols = lfp_img_.cols;

	// 定义通用 AWB 处理 lambda
	auto apply_awb = [&](auto *ptr_base, float max_val) {
		for (int r = 0; r < rows; ++r) {
			auto *ptr = ptr_base + r * cols; // 手动计算行指针（避免 .ptr<T>()）
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
				ptr[c] = static_cast<std::decay_t<decltype(ptr[c])>>(val);
			}
		}
	};

	if (lfp_img_.depth() == CV_16U) {
		apply_awb(lfp_img_.ptr<uint16_t>(), 65535.0f);
	} else if (lfp_img_.depth() == CV_8U) {
		apply_awb(lfp_img_.ptr<uint8_t>(), 255.0f);
	}

	return *this;
} // awb

LFIsp &LFIsp::demosaic(BayerPattern bayer) {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.channels() != 1)
		return *this;

	int code = get_demosaic_code(bayer, false);
	cv::demosaicing(lfp_img_, lfp_img_, code);
	return *this;
} // demosaic

LFIsp &LFIsp::resample(bool dehex) {
	int num_views = maps.extract.size() / 2;
	sais.clear();
	sais.resize(num_views);

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_views; ++i) {
		cv::Mat temp;
		cv::remap(lfp_img_, temp, maps.extract[i * 2], maps.extract[i * 2 + 1], cv::INTER_LANCZOS4,
				  cv::BORDER_REPLICATE);
		if (dehex && !maps.dehex.empty()) {
			cv::remap(temp, temp, maps.dehex[0], maps.dehex[1], cv::INTER_LANCZOS4, cv::BORDER_REPLICATE);
		}
		sais[i] = temp;
	}
	return *this;
} // resample

LFIsp &LFIsp::ccm(const std::vector<float> &ccm_matrix) {
	if (sais.empty()) // 假设类成员名为 sais
		return *this;

	cv::Mat m(3, 3, CV_32F, (void *)ccm_matrix.data());

	for (int i = 0; i < (int)sais.size(); ++i) {
		if (sais[i].empty() || sais[i].channels() != 3)
			continue;
		cv::transform(sais[i], sais[i], m);
	}

	return *this;
} // ccm

LFIsp &LFIsp::gc(float gamma) {
	if (sais.empty())
		return *this;

	// 1. 参数初始化 (sRGB 标准默认使用 1/2.4)
	float inv_gamma = (gamma > 1e-5f) ? gamma : 0.416667f;

	// sRGB 分段常数
	const float low_threshold = 0.0031308f;
	const float low_slope = 12.92f;
	const float alpha = 0.055f;

	// 2. 遍历视角栈中的每一张图像 (sais 是 std::vector<cv::Mat>)
	for (int i = 0; i < (int)sais.size(); ++i) {
		cv::Mat &img = sais[i];
		if (img.empty())
			continue;

		int rows = img.rows;
		int cols = img.cols;
		int chans = img.channels();

		// 3. 逐像素遍历 (最直接的 CV 指针访问)
		for (int r = 0; r < rows; ++r) {
			unsigned char *ptr = img.ptr<unsigned char>(r);

			// 线性遍历所有通道像素
			for (int c = 0; c < cols * chans; ++c) {
				// a. 归一化到 0.0 - 1.0
				float norm = static_cast<float>(ptr[c]) * (1.0f / 255.0f);
				float res;

				// b. 执行分段计算逻辑 (对应 CUDA 内核 apply_gamma_device)
				if (norm <= low_threshold) {
					// 暗部线性段
					res = low_slope * norm;
				} else {
					// 亮部幂律压缩段
					res = (1.0f + alpha) * std::pow(norm, inv_gamma) - alpha;
				}

				// c. 反归一化并钳位到 0-255
				// 使用 cv::saturate_cast 保证数值鲁棒性，防止溢出
				ptr[c] = cv::saturate_cast<unsigned char>(res * 255.0f + 0.5f);
			}
		}
	}

	return *this;
}

LFIsp &LFIsp::csc() {
	if (sais.empty()) {
		return *this;
	}

	static bool is_ycrcb = false;
	int code = is_ycrcb ? cv::COLOR_YCrCb2BGR : cv::COLOR_BGR2YCrCb;

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < static_cast<int>(sais.size()); ++i) {
		if (sais[i].empty() || sais[i].channels() != 3) {
			continue;
		}

		cv::cvtColor(sais[i], sais[i], code);
	}
	is_ycrcb = !is_ycrcb;

	return *this;
}

LFIsp &LFIsp::uvnr(float h_sigma_s, float h_sigma_r) {
	// 1. 基础检查：确保视角栈不为空
	if (sais.empty())
		return *this;

	// 2. 逐视角处理 (Baseline 版本不使用 OpenMP 加速)
	for (int i = 0; i < (int)sais.size(); ++i) {
		if (sais[i].empty() || sais[i].channels() != 3)
			continue;

		// 3. 拆分 YUV 通道
		std::vector<cv::Mat> yuv_channels;
		cv::split(sais[i], yuv_channels);

		// 4. 对色度通道 U 和 V 分别应用双边滤波
		cv::Mat filtered_u, filtered_v;

		cv::bilateralFilter(yuv_channels[1], filtered_u, 0, h_sigma_r, h_sigma_s);
		cv::bilateralFilter(yuv_channels[2], filtered_v, 0, h_sigma_r, h_sigma_s);

		// 5. 将处理后的色度数据回填
		yuv_channels[1] = filtered_u;
		yuv_channels[2] = filtered_v;

		// 6. 合并回三通道图像
		cv::merge(yuv_channels, sais[i]);
	}

	return *this;
} // uvnr

LFIsp &LFIsp::color_eq(ColorEqualizeMethod method) {
	if (lfp_img_.empty())
		return *this;

	ColorMatcher::equalize(sais, method);

	return *this;
}

LFIsp &LFIsp::ce(float clipLimit, int gridSize) {
	if (sais.empty())
		return *this;

	for (int i = 0; i < static_cast<int>(sais.size()); ++i) {
		if (sais[i].empty() || sais[i].channels() != 3)
			continue;

		std::vector<cv::Mat> channels;
		cv::split(sais[i], channels);

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(clipLimit);
		clahe->setTilesGridSize(cv::Size(gridSize, gridSize));
		clahe->apply(channels[0], channels[0]);
		cv::merge(channels, sais[i]);
	}

	return *this;
} // ce

LFIsp &LFIsp::se(float factor) {
	// 因子为 1.0 时不处理
	if (sais.empty() || std::abs(factor - 1.0f) < 1e-4)
		return *this;

	for (int i = 0; i < static_cast<int>(sais.size()); ++i) {
		if (sais[i].empty() || sais[i].channels() != 3)
			continue;

		std::vector<cv::Mat> channels;
		cv::split(sais[i], channels);

		// 计算色度偏移的中心点 (8-bit: 128, 16-bit: 32768)
		double shift = (sais[i].depth() == CV_8U) ? 128.0 : 32768.0;

		for (int k = 1; k <= 2; ++k) { // 处理 Cr (1) 和 Cb (2)
			cv::Mat float_chan;
			channels[k].convertTo(float_chan, CV_32F);

			// 饱和度计算公式：(Val - Mid) * Factor + Mid
			// 结果增加 0.5f 以确保 convertTo 时四舍五入准确
			float_chan = (float_chan - (float)shift) * factor + (float)shift;

			// 回填并截断
			float_chan.convertTo(channels[k], sais[i].type());
		}

		cv::merge(channels, sais[i]);
	}

	return *this;
}

LFIsp &LFIsp::process(const IspConfig &config) {
	ScopedTimer t_total(" Total Process", profiler_cpu, config.benchmark);

	if (lfp_img_.empty()) {
		std::cerr << "[LFISP] Cancelled: Cancelled: No source image available.";
		return *this;
	}

	{
		ScopedTimer t("BLC", profiler_cpu, config.benchmark);
		if (config.enableBLC) {
			blc(config.black_level, config.white_level);
		} else {
			std::cout << "[LFISP] Pipeline: 'BLC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Convert", profiler_cpu, config.benchmark);
		if (lfp_img_gpu.depth() != CV_8U) {
			lfp_img_gpu.convertTo(lfp_img_gpu, CV_8U, 255.0 / ((1 << config.bitDepth) - 1));
		}
	}
	{
		ScopedTimer t("DPC", profiler_cpu, config.benchmark);
		if (config.enableDPC) {
			dpc(config.dpcThreshold);
		} else {
			std::cout << "[LFISP] Pipeline: 'DPC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("NR", profiler_cpu, config.benchmark);
		if (config.enableRawNR) {
			rawnr(config.rawnr_sigma_s, config.rawnr_sigma_r);
		} else {
			std::cout << "[LFISP] Pipeline: 'NR' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("LSC", profiler_cpu, config.benchmark);
		if (config.enableLSC) {
			lsc(config.lscExp);
		} else {
			std::cout << "[LFISP] Pipeline: 'LSC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("AWB", profiler_cpu, config.benchmark);
		if (config.enableAWB) {
			awb(config.awb_gains);
		} else {
			std::cout << "[LFISP] Pipeline: 'AWB' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Demosaic", profiler_cpu, config.benchmark);
		if (config.enableDemosaic) {
			demosaic(config.bayer);
		} else {
			std::cout << "[LFISP] Pipeline: 'Demosaic' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Resample", profiler_cpu, config.benchmark);
		if (config.enableExtract) {
			resample(config.enableDehex);
			if (!config.enableDehex) {
				std::cout << "[LFISP] Pipeline: 'Dehex' is disabled in "
							 "settings."
						  << std::endl;
			}
		} else {
			std::cout << "[LFISP] Pipeline: 'Extract' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("CCM", profiler_cpu, config.benchmark);
		if (config.enableCCM) {
			ccm(config.ccm_matrix);
		} else {
			std::cout << "[LFISP] Pipeline: 'CCM' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Gamma", profiler_cpu, config.benchmark);
		if (config.enableGamma) {
			gc(config.gamma);
		} else {
			std::cout << "[LFISP] Pipeline: 'Gamma correction' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("RGB2YUV", profiler_cpu, config.benchmark);
		if (config.enableCSC) {
			csc();
		} else {
			std::cout << "[LFISP] Pipeline: 'RGB to YUV conversion' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("UVNR", profiler_cpu, config.benchmark);
		if (config.enableUVNR) {
			uvnr(config.uvnr_sigma_s, config.uvnr_sigma_r);
		} else {
			std::cout << "[LFISP] Pipeline: 'UVNR' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("Color Equalization", profiler_cpu, config.benchmark);
		if (config.enableColorEq) {
			color_eq(config.colorEqMethod);
		} else {
			std::cout << "[LFISP] Pipeline: 'Color Equalization' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("CE", profiler_cpu, config.benchmark);
		if (config.enableCE) {
			ce(config.ceClipLimit, config.ceGridSize);
		} else {
			std::cout << "[LFISP] Pipeline: 'CLAHE' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("SE", profiler_cpu, config.benchmark);
		if (config.enableSE) {
			se(config.seFactor);
		} else {
			std::cout << "[LFISP] Pipeline: 'SE' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("YUV2RGB", profiler_cpu, config.benchmark);
		if (config.enableCSC) {
			csc();
		} else {
			std::cout << "[LFISP] Pipeline: 'YUV to RGB conversion' is disabled in "
						 "settings."
					  << std::endl;
		}
	}

	return *this;
} // process

// ============================================================================
// 快速处理流程 (SIMD Implementation Dispatcher)
// ============================================================================

LFIsp &LFIsp::blc_fast(int black_level, int white_level) {
	if (lfp_img_.empty())
		return *this;

	// 1. 计算系数 (与 GPU 版本完全一致)
	double bl = static_cast<double>(black_level);
	double wl = static_cast<double>(white_level);
	double effective_range = wl - bl;
	if (effective_range < 1.0)
		effective_range = 1.0;

	float alpha = static_cast<float>(255.0 / effective_range);
	float beta = static_cast<float>(-bl * alpha);

	// 2. 准备 8-bit 输出矩阵 (不再原地修改，因为位深变了)
	cv::Mat dst(lfp_img_.rows, lfp_img_.cols, CV_8U);

	// 3. 展平处理 (如果是连续内存，视为一行处理，减少循环开销)
	// 注意：create 出来的 dst 可能会有 padding，需检查 isContinuous
	int rows = lfp_img_.rows;
	int cols = lfp_img_.cols;
	bool is_cont = lfp_img_.isContinuous() && dst.isContinuous();

	if (is_cont) {
		cols = rows * cols;
		rows = 1;
	}

	// 4. 并行计算
	if (lfp_img_.depth() == CV_16U) {
#pragma omp parallel for
		for (int r = 0; r < rows; ++r) {
			const uint16_t *src_ptr = lfp_img_.ptr<uint16_t>(r);
			uint8_t *dst_ptr = dst.ptr<uint8_t>(r);
			row_simd_u16_to_u8(src_ptr, dst_ptr, cols, alpha, beta);
		}
	} else if (lfp_img_.depth() == CV_8U) {
#pragma omp parallel for
		for (int r = 0; r < rows; ++r) {
			const uint8_t *src_ptr = lfp_img_.ptr<uint8_t>(r);
			uint8_t *dst_ptr = dst.ptr<uint8_t>(r);
			row_simd_u8_to_u8(src_ptr, dst_ptr, cols, alpha, beta);
		}
	}

	// 5. 替换原图
	lfp_img_ = dst;

	return *this;
} // blc_fast

LFIsp &LFIsp::dpc_fast(DpcMethod method, int threshold) {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.depth() == CV_16U) {
		dpc_simd_u16(lfp_img_, threshold);
	} else if (lfp_img_.depth() == CV_8U) {
		dpc_simd_u8(lfp_img_, threshold);
	}
	return *this;
} // dpc_fast

LFIsp &LFIsp::rawnr_fast(float sigma_spatial, float sigma_color) {
	if (lfp_img_.empty())
		return *this;

	// 限制 sigma 范围防止溢出
	sigma_spatial = std::max(0.5f, sigma_spatial);
	sigma_color = std::max(0.5f, sigma_color);

	if (lfp_img_.depth() == CV_16U) {
		nr_simd_u16(lfp_img_, sigma_spatial, sigma_color);
	} else if (lfp_img_.depth() == CV_8U) {
		nr_simd_u8(lfp_img_, sigma_spatial, sigma_color);
	}

	return *this;
}

LFIsp &LFIsp::lsc_fast(float exposure) {
	if (lfp_img_.empty())
		return *this;

	if (lsc_gain_map_int_.empty()) {
		return *this;
	}

	if (lfp_img_.depth() == CV_16U) {
		lsc_simd_u16(lfp_img_, lsc_gain_map_int_, exposure);
	} else if (lfp_img_.depth() == CV_8U) {
		lsc_simd_u8(lfp_img_, lsc_gain_map_int_, exposure);
	}

	return *this;
} // lsc_fast

LFIsp &LFIsp::awb_fast(const std::vector<float> &wbgains) {
	if (lfp_img_.empty())
		return *this;
	if (lfp_img_.depth() == CV_16U) {
		awb_simd_u16(lfp_img_, wbgains);
	} else if (lfp_img_.depth() == CV_8U) {
		awb_simd_u8(lfp_img_, wbgains);
	}
	return *this;
} // awb_fast

LFIsp &LFIsp::lsc_awb_fused_fast(float exposure, const std::vector<float> &wbgains) {
	if (lfp_img_.empty())
		return *this;

	if (lsc_gain_map_int_.empty()) {
		return *this;
	}

	if (lfp_img_.depth() == CV_16U) {
		lsc_awb_simd_u16(lfp_img_, exposure, lsc_gain_map_int_, wbgains);
	} else if (lfp_img_.depth() == CV_8U) {
		lsc_awb_simd_u8(lfp_img_, exposure, lsc_gain_map_int_, wbgains);
	}

	return *this;
} // lsc_awb_fused_fast

LFIsp &LFIsp::ccm_fast(const std::vector<float> &ccm_matrix) {
	if (sais.empty())
		return *this;

	// 1. 预处理固定点矩阵（在并行循环外完成，确保线程安全）
	if (ccm_matrix_int_.empty() || ccm_matrix != last_ccm_matrix_) {
		prepare_ccm_fixed_point(ccm_matrix);
		last_ccm_matrix_ = ccm_matrix;
	}

// 2. 使用 OpenMP 并行处理 81 个视角
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < (int)sais.size(); ++i) {
		cv::Mat &img = sais[i];
		if (img.empty() || img.channels() != 3)
			continue;

		// 根据位深调用对应的固定点运算函数
		if (img.depth() == CV_16U) {
			ccm_fixed_u16(img, ccm_matrix_int_);
		} else if (img.depth() == CV_8U) {
			ccm_fixed_u8(img, ccm_matrix_int_);
		}
	}
	return *this;
} // ccm_fast

LFIsp &LFIsp::gc_fast(float gamma, int bitDepth) {
	if (sais.empty())
		return *this;

	// 1. 状态检查与 LUT 准备（在并行循环外完成）
	bool gamma_changed = std::abs(gamma - last_gamma_) > 1e-6f;
	bool bitdepth_changed = (bitDepth != last_bit_depth_);

	// 预先根据第一个有效视角的位深准备好 LUT
	int depth = sais[0].depth();
	if (depth == CV_8U) {
		if (gamma_lut_u8.empty() || gamma_changed) {
			prepare_gamma_lut(gamma, 8);
			last_gamma_ = gamma;
			last_bit_depth_ = 8;
		}
	} else if (depth == CV_16U) {
		if (gamma_lut_u16.empty() || gamma_changed || bitdepth_changed) {
			prepare_gamma_lut(gamma, bitDepth);
			last_gamma_ = gamma;
			last_bit_depth_ = bitDepth;
		}
	}

// 2. 并行查表处理
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < (int)sais.size(); ++i) {
		cv::Mat &img = sais[i];
		if (img.empty())
			continue;

		if (img.depth() == CV_8U) {
			cv::LUT(img, gamma_lut_u8, img);
		} else if (img.depth() == CV_16U) {
			// 注意：OpenCV 的 cv::LUT 仅支持 8位输入，
			// 16位 Gamma 通常需要自定义映射函数或在 prepare_gamma_lut 中特殊处理
			cv::LUT(img, gamma_lut_u16, img);
		}
	}

	return *this;
}

LFIsp &LFIsp::uvnr_fast(float h_sigma_s, float h_sigma_r) {
	if (sais.empty()) {
		return *this;
	}

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < static_cast<int>(sais.size()); ++i) {
		cv::Mat &img = sais[i];

		// 确保输入是 3 通道 YUV 数据 (CV_8UC3)
		if (img.empty() || img.channels() != 3) {
			continue;
		}

		// 1. 拆分 YUV 通道
		std::vector<cv::Mat> yuv_channels;
		cv::split(img, yuv_channels);

		// 2. 对 U 和 V 分量应用双边滤波
		// Y 通道 (Index 0) 包含核心结构信息，保持不动以防模糊
		cv::Mat filtered_u, filtered_v;

		// d=0 表示滤波器直径由 sigma_s 自动计算
		cv::bilateralFilter(yuv_channels[1], filtered_u, 0, h_sigma_r, h_sigma_s);
		cv::bilateralFilter(yuv_channels[2], filtered_v, 0, h_sigma_r, h_sigma_s);

		// 3. 回填降噪后的色度分量
		yuv_channels[1] = filtered_u;
		yuv_channels[2] = filtered_v;

		// 4. 合并回原视图
		cv::merge(yuv_channels, img);
	}

	return *this;
}

LFIsp &LFIsp::ce_fast(float clipLimit, int gridSize) {
	if (sais.empty())
		return *this;

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < static_cast<int>(sais.size()); ++i) {
		if (sais[i].empty() || sais[i].channels() != 3)
			continue;

		std::vector<cv::Mat> channels(3);
		cv::split(sais[i], channels);

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(clipLimit);
		clahe->setTilesGridSize(cv::Size(gridSize, gridSize));
		clahe->apply(channels[0], channels[0]);
		cv::merge(channels, sais[i]);
	}

	return *this;
} // ce_fast

LFIsp &LFIsp::se_fast(float factor) {
	if (sais.empty())
		return *this;

	// 1. 预计算 3 组不同的掩码，对应 SIMD 块相对于 YUV 三元组的 3 种相位
	int16_t f = static_cast<int16_t>(factor * 256.0f);
	__m256i v_offset = _mm256_set1_epi16(128);

	// 相位 0: [Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y]
	__m256i v_f0 = _mm256_setr_epi16(256, f, f, 256, f, f, 256, f, f, 256, f, f, 256, f, f, 256);
	// 相位 1: [Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr]
	__m256i v_f1 = _mm256_setr_epi16(f, f, 256, f, f, 256, f, f, 256, f, f, 256, f, f, 256, f);
	// 相位 2: [Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb, Y, Cr, Cb]
	__m256i v_f2 = _mm256_setr_epi16(f, 256, f, f, 256, f, f, 256, f, f, 256, f, f, 256, f, f);

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < static_cast<int>(sais.size()); ++i) {
		cv::Mat &img = sais[i];
		if (img.empty() || img.channels() != 3)
			continue;

		for (int r = 0; r < img.rows; ++r) {
			uint8_t *ptr = img.ptr<uint8_t>(r);
			int c = 0;
			int total_channels = img.cols * 3;

			// 2. 以 48 字节为一组处理，消除相位偏移
			for (; c <= total_channels - 48; c += 48) {
				// 处理第一个 16 字节 (相位 0)
				__m256i v0 = _mm256_add_epi16(
					_mm256_srai_epi16(
						_mm256_mullo_epi16(
							_mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(ptr + c))), v_offset),
							v_f0),
						8),
					v_offset);
				__m256i p0 = _mm256_permute4x64_epi64(_mm256_packus_epi16(v0, v0), 0xD8);
				_mm_storeu_si128((__m128i *)(ptr + c), _mm256_castsi256_si128(p0));

				// 处理第二个 16 字节 (相位 1)
				__m256i v1 = _mm256_add_epi16(
					_mm256_srai_epi16(
						_mm256_mullo_epi16(
							_mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(ptr + c + 16))),
											 v_offset),
							v_f1),
						8),
					v_offset);
				__m256i p1 = _mm256_permute4x64_epi64(_mm256_packus_epi16(v1, v1), 0xD8);
				_mm_storeu_si128((__m128i *)(ptr + c + 16), _mm256_castsi256_si128(p1));

				// 处理第三个 16 字节 (相位 2)
				__m256i v2 = _mm256_add_epi16(
					_mm256_srai_epi16(
						_mm256_mullo_epi16(
							_mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(ptr + c + 32))),
											 v_offset),
							v_f2),
						8),
					v_offset);
				__m256i p2 = _mm256_permute4x64_epi64(_mm256_packus_epi16(v2, v2), 0xD8);
				_mm_storeu_si128((__m128i *)(ptr + c + 32), _mm256_castsi256_si128(p2));
			}

			// 3. 处理余下的末尾字节
			for (; c < total_channels; ++c) {
				if (c % 3 != 0) { // 跳过 Y 通道
					ptr[c] = cv::saturate_cast<uint8_t>((ptr[c] - 128) * factor + 128);
				}
			}
		}
	}
	return *this;
} // se_fast

LFIsp &LFIsp::preview(const IspConfig &config) {
	if (lfp_img_.empty() || config.bayer == BayerPattern::NONE)
		return *this;

	int rows = lfp_img_.rows;
	int cols = lfp_img_.cols;
	cv::Mat raw_8u(rows, cols, CV_8UC1);

	if (lsc_gain_map_int_.empty()) {
		return *this;
	}

	// 根据位深分发
	if (lfp_img_.depth() == CV_16U) {
		raw_to_8bit_with_gains_simd_u16(lfp_img_, raw_8u, config, lsc_gain_map_int_);
	} else if (lfp_img_.depth() == CV_8U) {
		raw_to_8bit_with_gains_simd_u8(lfp_img_, raw_8u, config, lsc_gain_map_int_);
	}

	int code = get_demosaic_code(config.bayer, false);
	cv::demosaicing(raw_8u, lfp_img_, code);

	return *this;
} // preview

LFIsp &LFIsp::process_fast(const IspConfig &config) {
	ScopedTimer t_total("Total Process", profiler_fast, config.benchmark);

	if (lfp_img_.empty()) {
		std::cerr << "[LFISP] Cancelled: Cancelled: No source image available.";
		return *this;
	}

	{
		ScopedTimer t("BLC", profiler_fast, config.benchmark);
		if (config.enableBLC) {
			// blc_fast(config.black_level);
			blc_fast(config.black_level, config.white_level);
		} else {
			std::cout << "[LFISP] Pipeline: 'BLC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Convert", profiler_fast, config.benchmark);
		if (lfp_img_.depth() != CV_8U) {
			lfp_img_.convertTo(lfp_img_, CV_8U, 255.0 / ((1 << config.bitDepth) - 1));
		}
	}

	{
		ScopedTimer t("DPC", profiler_fast, config.benchmark);
		if (config.enableDPC) {
			dpc_fast(config.dpcMethod, config.dpcThreshold);
		} else {
			std::cout << "[LFISP] Pipeline: 'DPC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("NR", profiler_fast, config.benchmark);
		if (config.enableRawNR) {
			rawnr_fast(config.rawnr_sigma_s, config.rawnr_sigma_r);
		} else {
			std::cout << "[LFISP] Pipeline: 'NR' is disabled in settings." << std::endl;
		}
	}
	if (config.enableLSC && config.enableAWB) {
		ScopedTimer t("LSC+AWB", profiler_fast, config.benchmark);
		lsc_awb_fused_fast(config.lscExp, config.awb_gains);
	} else {
		{
			ScopedTimer t("LSC", profiler_fast, config.benchmark);
			if (config.enableLSC) {
				lsc_fast(config.lscExp);
			} else {
				std::cout << "[LFISP] Pipeline: 'LSC' is disabled in settings." << std::endl;
			}
		}
		{
			ScopedTimer t("AWB", profiler_fast, config.benchmark);
			if (config.enableAWB) {
				awb_fast(config.awb_gains);
			} else {
				std::cout << "[LFISP] Pipeline: 'AWB' is disabled in settings." << std::endl;
			}
		}
	}
	{
		ScopedTimer t("Demosaic", profiler_fast, config.benchmark);
		if (config.enableDemosaic) {
			demosaic(config.bayer);
		} else {
			std::cout << "[LFISP] Pipeline: 'Demosaic' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Resample", profiler_fast, config.benchmark);
		if (config.enableExtract && !maps.extract.empty()) {
			resample(config.enableDehex);
			if (!config.enableDehex) {
				std::cout << "[LFISP] Pipeline: 'Dehex' is disabled in "
							 "settings."
						  << std::endl;
			}
		} else {
			std::cout << "[LFISP] Pipeline: 'Extract' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("CCM", profiler_fast, config.benchmark);
		if (config.enableCCM) {
			ccm_fast(config.ccm_matrix);
		} else {
			std::cout << "[LFISP] Pipeline: 'CCM' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Gamma", profiler_fast, config.benchmark);
		if (config.enableGamma) {
			gc_fast(config.gamma, 8);
		} else {
			std::cout << "[LFISP] Pipeline: 'Gamma correction' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("RGB2YUV", profiler_cpu, config.benchmark);
		if (config.enableCSC) {
			csc();
		} else {
			std::cout << "[LFISP] Pipeline: 'RGB to YUV conversion' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("UVNR", profiler_cpu, config.benchmark);
		if (config.enableUVNR) {
			uvnr(config.uvnr_sigma_s, config.uvnr_sigma_r);
		} else {
			std::cout << "[LFISP] Pipeline: 'UVNR' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("Color Equalization", profiler_cpu, config.benchmark);
		if (config.enableColorEq) {
			color_eq(config.colorEqMethod);
		} else {
			std::cout << "[LFISP] Pipeline: 'Color Equalization' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("CE", profiler_cpu, config.benchmark);
		if (config.enableCE) {
			ce_fast(config.ceClipLimit, config.ceGridSize);
		} else {
			std::cout << "[LFISP] Pipeline: 'CLAHE' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("SE", profiler_cpu, config.benchmark);
		if (config.enableSE) {
			se_fast(config.seFactor);
		} else {
			std::cout << "[LFISP] Pipeline: 'SE' is disabled in "
						 "settings."
					  << std::endl;
		}
	}
	{
		ScopedTimer t("YUV2RGB", profiler_cpu, config.benchmark);
		if (config.enableCSC) {
			csc();
		} else {
			std::cout << "[LFISP] Pipeline: 'YUV to RGB conversion' is disabled in "
						 "settings."
					  << std::endl;
		}
	}

	return *this;
} // process_fast

void LFIsp::prepare_lsc_maps(const cv::Mat &raw_wht, int black_level) {
	int rows = raw_wht.rows;
	int cols = raw_wht.cols;

	cv::Mat float_wht;
	raw_wht.convertTo(float_wht, CV_32F);

	float bl = static_cast<float>(black_level);
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
		cv::GaussianBlur(channels[k], channels[k], cv::Size(9, 9), 0);
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
	lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, FIXED_SCALE);
	lsc_map_gpu.upload(lsc_gain_map_);
} // prepare_lsc_maps

// ============================================================================
// CCM SIMD Implementation
// ============================================================================

void LFIsp::prepare_ccm_fixed_point(const std::vector<float> &matrix) {
	if (matrix.empty())
		return;
	ccm_matrix_int_.resize(9);

	const float scale = FIXED_SCALE;

	for (int i = 0; i < 9; ++i) {
		ccm_matrix_int_[i] = static_cast<int32_t>(matrix[i] * scale + 0.5f);
	}
	return;
} // prepare_ccm_fixed_point

void LFIsp::prepare_gamma_lut(float gamma, int bitDepth) {
	// 默认使用 sRGB 标准指数 1/2.4 ≈ 0.416667
	float g = gamma > 0 ? gamma : 0.416667f;

	// sRGB 标准分段函数参数
	const double alpha = 0.055;
	const double low_threshold = 0.0031308;
	const double low_slope = 12.92;

	if (bitDepth > 8) {
		if (gamma_lut_u16.size() != 65536) {
			gamma_lut_u16.resize(65536);
		}
		if (!gamma_lut_u8.empty()) {
			gamma_lut_u8.release();
		}

		double max_input_val = (1 << bitDepth) - 1.0;
		double max_output_val = 65535.0;

#pragma omp parallel for
		for (int i = 0; i < 65536; ++i) {
			if (i > max_input_val) {
				gamma_lut_u16[i] = (uint16_t)max_output_val;
				continue;
			}

			double norm = (double)i / max_input_val;
			double res;

			// 分段逻辑：暗部线性拉伸，亮部非线性压缩
			if (norm <= low_threshold) {
				res = low_slope * norm;
			} else {
				res = (1.0 + alpha) * std::pow(norm, g) - alpha;
			}

			gamma_lut_u16[i] = cv::saturate_cast<uint16_t>(res * max_output_val);
		}
	} else {
		if (!gamma_lut_u16.empty()) {
			gamma_lut_u16.clear();
		}
		if (gamma_lut_u8.empty()) {
			gamma_lut_u8.create(1, 256, CV_8U);
		}

		uchar *p = gamma_lut_u8.ptr();
		for (int i = 0; i < 256; ++i) {
			double norm = i / 255.0;
			double res;

			if (norm <= low_threshold) {
				res = low_slope * norm;
			} else {
				res = (1.0 + alpha) * std::pow(norm, g) - alpha;
			}

			p[i] = cv::saturate_cast<uchar>(res * 255.0);
		}
	}
}

//  GPU
LFIsp &LFIsp::set_lf_gpu(const cv::Mat &img) {
	if (img.empty()) {
		return *this;
	}
	lfp_img_gpu.upload(img);
	return *this;
} // set_lf_gpu

cv::Mat LFIsp::getResultGpu() {
	cv::Mat temp;
	lfp_img_gpu.download(temp);
	temp.convertTo(temp, CV_8U);
	return temp;
} // getResultGpu

std::vector<cv::Mat> LFIsp::getSAIsGpu() {
	std::vector<cv::Mat> views(sais_gpu.size());
	for (int i = 0; i < sais_gpu.size(); ++i) {
		sais_gpu[i].download(views[i]);
	}
	return views;
} // getSAIsGpu

void LFIsp::update_resample_maps() {
	extract_maps_gpu.clear();
	extract_maps_gpu.resize(maps.extract.size());
	for (int i = 0; i < maps.extract.size(); ++i) {
		extract_maps_gpu[i].upload(maps.extract[i]);
	}
	dehex_maps_gpu.clear();
	dehex_maps_gpu.resize(maps.dehex.size());
	for (int i = 0; i < maps.dehex.size(); ++i) {
		dehex_maps_gpu[i].upload(maps.dehex[i]);
	}
} // update_resample_maps

LFIsp &LFIsp::blc_gpu(int black_level, int white_level) {
	if (lfp_img_gpu.empty()) {
		return *this;
	}
	double bl = static_cast<double>(black_level);
	double wl = static_cast<double>(white_level);

	double effective_range = wl - bl;
	if (effective_range < 1.0)
		effective_range = 1.0; // 防止除零

	double alpha = 255.0 / effective_range; // 缩放系数
	double beta = -bl * alpha;				// 偏移系数

	lfp_img_gpu.convertTo(lfp_img_gpu, CV_8U, alpha, beta, stream);
	return *this;
} // blc_gpu

LFIsp &LFIsp::dpc_gpu(int threshold) {
	if (lfp_img_gpu.empty())
		return *this;

	launch_dpc_8u_inplace(lfp_img_gpu, threshold, stream);

	return *this;
} // dpc_gpu

LFIsp &LFIsp::nr_gpu(float sigma_spatial, float sigma_color) {
	if (lfp_img_gpu.empty())
		return *this;

	launch_nr_8u(lfp_img_gpu, lfp_img_gpu, sigma_spatial, sigma_color, stream);

	return *this;
}

LFIsp &LFIsp::lsc_gpu(float exposure) {
	// 检查非空
	if (lfp_img_gpu.empty() || lsc_map_gpu.empty())
		return *this;

	launch_lsc_8u_apply_32f(lfp_img_gpu, lsc_map_gpu, exposure, stream);

	return *this;
} // lsc_gpu

LFIsp &LFIsp::awb_gpu(const std::vector<float> &wbgains) {
	if (lfp_img_gpu.empty())
		return *this;

	if (wbgains.size() < 4)
		return *this;

	launch_awb_8u(lfp_img_gpu, wbgains[0], wbgains[1], wbgains[2], wbgains[3], stream);

	return *this;
} // awb_gpu

LFIsp &LFIsp::lsc_awb_fused_gpu(float exposure, const std::vector<float> &wbgains) {
	// 1. 基础检查
	if (lfp_img_gpu.empty() || lsc_map_gpu.empty())
		return *this;
	if (lfp_img_gpu.type() != CV_8UC1)
		lfp_img_gpu.convertTo(lfp_img_gpu, CV_8U); // 确保 8U
	if (lsc_map_gpu.type() != CV_32FC1)
		return *this; // Map 必须是 Float
	if (wbgains.size() < 4)
		return *this;

	// 3. 调用融合 Kernel
	launch_fused_lsc_awb(lfp_img_gpu, lsc_map_gpu, exposure, wbgains[0], wbgains[1], wbgains[2], wbgains[3], stream);

	return *this;
} // lsc_awb_fused_gpu

LFIsp &LFIsp::demosaic_gpu(BayerPattern bayer) {
	if (lfp_img_gpu.empty())
		return *this;

	if (lfp_img_gpu.depth() != CV_8U) {
		cv::cuda::GpuMat temp;
		lfp_img_gpu.convertTo(temp, CV_8U);
		lfp_img_gpu = temp;
	}

	int code = -1;
	switch (bayer) {
		case BayerPattern::RGGB:
			// 第一行 RG -> Input=BayerRG
			code = cv::cuda::COLOR_BayerRG2BGR_MHT;
			break;

		case BayerPattern::GRBG:
			// 第一行 GR -> Input=BayerGR (你之前写成了 GB，这是错的)
			code = cv::cuda::COLOR_BayerGR2BGR_MHT;
			break;

		case BayerPattern::GBRG:
			// 第一行 GB -> Input=BayerGB (你之前写成了 GR，这是错的)
			code = cv::cuda::COLOR_BayerGB2BGR_MHT;
			break;

		case BayerPattern::BGGR:
			// 第一行 BG -> Input=BayerBG
			code = cv::cuda::COLOR_BayerBG2BGR_MHT;
			break;

		default:
			return *this;
	}

	if (code >= 0) {
		cv::cuda::GpuMat dst_bgr;
		cv::cuda::demosaicing(lfp_img_gpu, dst_bgr, code);
		lfp_img_gpu = dst_bgr;
	}

	return *this;
} // demosaic_gpu

LFIsp &LFIsp::resample_gpu(bool dehex) {
	if (lfp_img_gpu.empty()) {
		return *this;
	}

	if (maps.extract.empty()) {
		return *this;
	} else if (extract_maps_gpu.empty()) {
		update_resample_maps();
	}

	int num_views = extract_maps_gpu.size() / 2;
	sais_gpu.clear();
	sais_gpu.resize(num_views);

	for (int i = 0; i < num_views; ++i) {
		cv::cuda::GpuMat extracted;
		cv::cuda::remap(lfp_img_gpu, extracted, extract_maps_gpu[2 * i], extract_maps_gpu[2 * i + 1], cv::INTER_LINEAR,
						cv::BORDER_REPLICATE);

		if (dehex && !dehex_maps_gpu.empty()) {
			cv::cuda::remap(extracted, sais_gpu[i], dehex_maps_gpu[0], dehex_maps_gpu[1], cv::INTER_LINEAR,
							cv::BORDER_REPLICATE);
		} else {
			sais_gpu[i] = extracted;
		}
	}
	return *this;
} // resample_gpu

LFIsp &LFIsp::ccm_gpu(const std::vector<float> &ccm_matrix) {
	if (lfp_img_gpu.empty() || lfp_img_gpu.type() != CV_8UC3) {
		std::cout << "[LFISP] CCM: Invalid image type." << std::endl;
		return *this;
	}

	if (ccm_matrix.size() < 9) {
		std::cout << "[LFISP] CCM: Invalid ccm matrix." << std::endl;
		return *this;
	}

	for (size_t i = 0; i < sais_gpu.size(); ++i) {
		if (sais_gpu[i].empty() || sais_gpu[i].type() != CV_8UC3) {
			continue;
		}
		launch_ccm_8uc3(sais_gpu[i], ccm_matrix.data(), stream);
	}

	return *this;
} // ccm_gpu

LFIsp &LFIsp::gc_gpu(float gamma) {
	// 1. 检查 SAI 向量是否为空
	if (sais_gpu.empty())
		return *this;

	// 2. 检查参数有效性
	// 如果 Gamma 接近 1.0，通常视为线性输出，不执行操作
	if (gamma > 0 && std::abs(gamma - 1.0f) < 1e-5)
		return *this;

	// 3. 遍历所有视角图像
	for (size_t i = 0; i < sais_gpu.size(); ++i) {
		if (sais_gpu[i].empty() || sais_gpu[i].depth() != CV_8U) {
			continue;
		}
		launch_gc_8u(sais_gpu[i], gamma, stream);
	}

	return *this;
} // gc_gpu

LFIsp &LFIsp::ccm_gamma_fused_gpu(const std::vector<float> &ccm_matrix, float gamma) {
	// 1. 检查输入：必须是 8UC3 (Demosaic 之后的图)
	if (lfp_img_gpu.empty() || lfp_img_gpu.type() != CV_8UC3) {
		return *this;
	}

	// 2. 检查矩阵维度
	if (ccm_matrix.size() < 9) {
		return *this;
	}

	// 3. 处理 Gamma 值
	// 如果你想变亮 (标准 Gamma 2.2)，这里应该传入 1.0/2.2 ≈ 0.45
	// 如果你在 Config 里存的是 2.2，就在这里取倒数
	// 如果你在 Config 里存的是 0.45，就直接传
	float gamma_val = (std::abs(gamma) > 1e-5) ? gamma : 1.0f;

	// 如果你的 config.gamma 是 2.2，建议用下面这行：
	// float gamma_val = 1.0f / gamma;

	// 4. 调用融合 Kernel
	launch_ccm_gamma_fused(lfp_img_gpu, ccm_matrix.data(), gamma_val, stream);

	return *this;
} // ccm_gamma_fused_gpu

LFIsp &LFIsp::csc_gpu() {
	if (sais_gpu.empty())
		return *this;

	static bool is_ycrcb_gpu = false;
	int code = !is_ycrcb_gpu ? cv::COLOR_BGR2YCrCb : cv::COLOR_YCrCb2BGR;

	// 遍历所有视角图像
	for (size_t i = 0; i < sais_gpu.size(); ++i) {
		if (sais_gpu[i].empty())
			continue;

		cv::cuda::cvtColor(sais_gpu[i], sais_gpu[i], code, 0, stream);
	}

	is_ycrcb_gpu = !is_ycrcb_gpu;

	return *this;
}

LFIsp &LFIsp::uvnr_gpu(float h_sigma_s, float h_sigma_r) {
	if (sais_gpu.empty())
		return *this;

	// 参数合理性检查
	if (h_sigma_s < 0.1f || h_sigma_r < 0.1f)
		return *this;

	for (size_t i = 0; i < sais_gpu.size(); ++i) {
		if (sais_gpu[i].empty() || sais_gpu[i].type() != CV_8UC3) {
			continue;
		}

		// 调用手写高性能内核
		launch_uvnr_8uc3(sais_gpu[i], h_sigma_s, h_sigma_r, stream);
	}

	return *this;
}

LFIsp &LFIsp::color_eq_gpu(ColorEqualizeMethod method) {
	if (sais_gpu.empty())
		return *this;

	ColorMatcher::equalize_gpu(sais_gpu, method, stream);
	ColorMatcher::equalize_gpu(sais_gpu, method, stream);
	ColorMatcher::equalize_gpu(sais_gpu, method, stream);

	return *this;
}

LFIsp &LFIsp::ce_gpu(float clipLimit, int gridSize) {
	if (lfp_img_gpu.empty())
		return *this;

	cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(clipLimit, cv::Size(gridSize, gridSize));
	for (size_t i = 0; i < sais_gpu.size(); ++i) {
		if (sais_gpu[i].empty() || sais_gpu[i].channels() != 3)
			continue;

		// 1. 分离通道 (GPU 端)
		// 注意：建议预分配 channels 缓存以进一步压榨性能
		std::vector<cv::cuda::GpuMat> channels;
		cv::cuda::split(sais_gpu[i], channels, stream);

		// 2. 仅对 Y 通道 (index 0) 应用 CLAHE
		//
		clahe->apply(channels[0], channels[0], stream);

		// 3. 合并回原图
		cv::cuda::merge(channels, sais_gpu[i], stream);
	}

	return *this;
}

LFIsp &LFIsp::se_gpu(float factor) {
	if (lfp_img_gpu.empty() || std::abs(factor - 1.0f) < 1e-4)
		return *this;

	for (size_t i = 0; i < sais_gpu.size(); ++i) {
		if (sais_gpu[i].empty())
			continue;
		launch_se_gpu(sais_gpu[i], factor, stream);
	}

	return *this;
}

LFIsp &LFIsp::process_gpu(const IspConfig &config) {
	ScopedTimer t_total(" Total Process", profiler_gpu, config.benchmark);

	if (lfp_img_gpu.empty()) {
		std::cerr << "[LFISP] Cancelled: Cancelled: No source image available.";
		return *this;
	}
	{
		ScopedTimer t("BLC", profiler_gpu, config.benchmark);
		if (config.enableBLC) {
			blc_gpu(config.black_level, config.white_level);
		} else {
			std::cout << "[LFISP] Pipeline: 'BLC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Convert", profiler_gpu, config.benchmark);
		if (lfp_img_gpu.depth() != CV_8U) {
			lfp_img_gpu.convertTo(lfp_img_gpu, CV_8U, 255.0 / ((1 << config.bitDepth) - 1));
		}
	}

	{
		ScopedTimer t("DPC", profiler_gpu, config.benchmark);
		if (config.enableDPC) {
			dpc_gpu(config.dpcThreshold);
		} else {
			std::cout << "[LFISP] Pipeline: 'DPC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Noise Reduction", profiler_gpu, config.benchmark);
		if (config.enableRawNR) {
			nr_gpu(config.rawnr_sigma_s, config.rawnr_sigma_r);
		} else {
			std::cout << "[LFISP] Pipeline: 'Noise Reduction' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("LSC", profiler_gpu, config.benchmark);
		if (config.enableLSC) {
			lsc_gpu(config.lscExp);
		} else {
			std::cout << "[LFISP] Pipeline: 'LSC' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("AWB", profiler_gpu, config.benchmark);
		if (config.enableAWB) {
			awb_gpu(config.awb_gains);
		} else {
			std::cout << "[LFISP] Pipeline: 'AWB' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Demosaic", profiler_gpu, config.benchmark);
		if (config.enableDemosaic) {
			demosaic_gpu(config.bayer);
		} else {
			std::cout << "[LFISP] Pipeline: 'Demosaic' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Resample", profiler_gpu, config.benchmark);
		if (config.enableExtract) {
			resample_gpu(config.enableDehex);
			if (!config.enableDehex) {
				std::cout << "[LFISP] Pipeline: 'Dehex' is disabled in settings." << std::endl;
			}
		} else {
			std::cout << "[LFISP] Pipeline: 'Extract' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("CCM", profiler_gpu, config.benchmark);
		if (config.enableCCM) {
			ccm_gpu(config.ccm_matrix);
		} else {
			std::cout << "[LFISP] Pipeline: 'CCM' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Gamma", profiler_gpu, config.benchmark);
		if (config.enableGamma) {
			gc_gpu(config.gamma);
		} else {
			std::cout << "[LFISP] Pipeline: 'Gamma correction' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("RGB2YUV", profiler_gpu, config.benchmark);
		if (config.enableCSC) {
			csc_gpu();
		} else {
			std::cout << "[LFISP] Pipeline: 'RGB to YUV' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("Color Equalization", profiler_gpu, config.benchmark);
		if (config.enableColorEq) {
			color_eq_gpu(config.colorEqMethod);
		} else {
			std::cout << "[LFISP] Pipeline: 'Color Equalization' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("UVNR", profiler_gpu, config.benchmark);
		if (config.enableUVNR) {
			uvnr_gpu(config.uvnr_sigma_s, config.uvnr_sigma_r);
		} else {
			std::cout << "[LFISP] Pipeline: 'UVNR' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("CE", profiler_gpu, config.benchmark);
		if (config.enableCE) {
			ce_gpu(config.ceClipLimit, config.ceGridSize);
		} else {
			std::cout << "[LFISP] Pipeline: 'CE' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("SE", profiler_gpu, config.benchmark);
		if (config.enableSE) {
			se_gpu(config.seFactor);
		} else {
			std::cout << "[LFISP] Pipeline: 'SE' is disabled in settings." << std::endl;
		}
	}
	{
		ScopedTimer t("YUV2RGB", profiler_gpu, config.benchmark);
		if (config.enableCSC) {
			csc_gpu();
		} else {
			std::cout << "[LFISP] Pipeline: 'YUV to RGB' is disabled in settings." << std::endl;
		}
	}

	return *this;
} // process_gpu

// --- lfisp.cpp ---

LFIsp &LFIsp::global_resample_gpu(std::shared_ptr<HexGridFitter> fitter, bool hex_stretch) {
	if (lfp_img_gpu.empty())
		return *this;

	HexGridFitter::GridInfo info = fitter->get_grid_info();
	// 1. 确定缩放比例 (Source -> Target)
	// 注意：H 是 Target -> Source 的映射，所以系数是 Pitch_Source / Pitch_Target
	float target_pitch_v = std::ceil(info.pitch_row / 2.0f) * 2.0f;
	float target_pitch_h = std::ceil(info.pitch_col / 2.0f) * 2.0f;

	float s_y = info.pitch_row / target_pitch_v;
	float s_x = info.pitch_col / target_pitch_h;
	if (hex_stretch)
		s_x *= (std::sqrt(3.0f) / 2.0f); // 补偿水平拉伸

	// 2. 计算输出画布尺寸：直接基于原图尺寸进行缩放
	cv::Size out_size(std::round(lfp_img_gpu.cols / s_x), std::round(lfp_img_gpu.rows / s_y));

	// 3. 构建旋转与缩放矩阵 (Target -> Source)
	// 提取拟合出的旋转分量 (通过参数归一化得到 cos/sin)
	float cos_theta = fitter->get_params_at(1, 0) / info.pitch_col; // a1 / pitch_h
	float sin_theta = fitter->get_params_at(1, 1) / info.pitch_col; // b1 / pitch_h

	cv::Mat H = cv::Mat::eye(3, 3, CV_32F);
	H.at<float>(0, 0) = s_x * cos_theta;
	H.at<float>(0, 1) = -s_y * sin_theta;
	H.at<float>(1, 0) = s_x * sin_theta;
	H.at<float>(1, 1) = s_y * cos_theta;

	// 4. 关键步骤：平移对齐（将源图像中心映射到目标画布中心）
	// ox = H00*tx + H01*ty + Bx
	// 我们希望当 tx = out_width/2, ty = out_height/2 时，ox = src_width/2
	float src_cx = lfp_img_gpu.cols / 2.0f;
	float src_cy = lfp_img_gpu.rows / 2.0f;
	float dst_cx = out_size.width / 2.0f;
	float dst_cy = out_size.height / 2.0f;

	H.at<float>(0, 2) = src_cx - (H.at<float>(0, 0) * dst_cx + H.at<float>(0, 1) * dst_cy);
	H.at<float>(1, 2) = src_cy - (H.at<float>(1, 0) * dst_cx + H.at<float>(1, 1) * dst_cy);

	// 5. 执行变换
	cv::cuda::GpuMat aligned_gpu;
	cv::cuda::warpPerspective(lfp_img_gpu, aligned_gpu, H, out_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
							  cv::Scalar(0), stream);

	lfp_img_gpu = aligned_gpu;
	return *this;
}