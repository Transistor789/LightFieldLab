#ifndef LFISP_H
#define LFISP_H

#include "json.hpp"
#include "lfparams.h"
#include "utils.h"

#include <immintrin.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using json = nlohmann::json;

class LFIsp {
public:
	struct ResampleMaps {
		std::vector<cv::Mat> extract;
		std::vector<cv::Mat> dehex;
	} maps;

public:
	explicit LFIsp();
	explicit LFIsp(const LFParamsSource &config, const cv::Mat &lfp_img,
				   const cv::Mat &wht_img);
	explicit LFIsp(const json &json_config, const cv::Mat &lfp_img,
				   const cv::Mat &wht_img);

	LFParamsSource getConfig() const { return config_; }
	cv::Mat &getResult() { return lfp_img_; }
	const cv::Mat &getPreviewResult() const { return preview_img_; }
	const cv::Mat &getResult() const { return lfp_img_; }
	const std::vector<cv::Mat> &getSAIS() const { return sais; }

	LFParamsSource &get_config() { return config_; }

	LFIsp &set_config(const LFParamsSource &new_config);
	LFIsp &set_config(const json &json_settings);
	LFIsp &print_config();

	LFIsp &updateImage(const cv::Mat &new_image, bool isWhite = false);
	LFIsp &set_lf_img(const cv::Mat &img);
	LFIsp &set_white_img(const cv::Mat &img);

	// 预览流程
	LFIsp &preview(float exposure = 1.5f);

	// 标量处理函数 (内部根据 bitDepth 分发)
	LFIsp &blc();
	LFIsp &dpc(int threshold = 100);
	LFIsp &lsc();
	LFIsp &awb();
	LFIsp &ccm();
	LFIsp &demosaic();
	LFIsp &gc(); // Gamma Correction placeholder

	LFIsp &raw_process();

	// SIMD 加速函数 (内部根据 bitDepth 分发)
	LFIsp &blc_fast();
	LFIsp &dpc_fast(int threshold = 100);
	LFIsp &lsc_fast();
	LFIsp &awb_fast();
	LFIsp &lsc_awb_fused_fast();
	LFIsp &ccm_fast();

	LFIsp &raw_process_fast();

	// 后处理
	LFIsp &resample(bool dehex);
	LFIsp &color_equalize();

private:
	// 成员变量
	LFParamsSource config_;
	cv::Mat preview_img_;
	cv::Mat lfp_img_, wht_img_;
	cv::Mat lsc_gain_map_, lsc_gain_map_int_;
	std::vector<int32_t> ccm_matrix_int_;
	std::vector<cv::Mat> sais;

	// 内部辅助函数
	std::string bayer_to_string(BayerPattern p) const;
	void generate_lsc_maps(const cv::Mat &raw_wht);
	LFIsp &prepare_ccm_fixed_point();

	LFIsp &compute_lab_stats(const cv::Mat &src, cv::Scalar &mean,
							 cv::Scalar &stddev);
	LFIsp &apply_reinhard_transfer(cv::Mat &target, const cv::Scalar &ref_mean,
								   const cv::Scalar &ref_std);

	// SIMD 实现的具体版本
	LFIsp &blc_simd_u16(cv::Mat &img);
	LFIsp &blc_simd_u8(cv::Mat &img);

	LFIsp &dpc_simd_u16(cv::Mat &img, int threshold);
	LFIsp &dpc_simd_u8(cv::Mat &img, int threshold);

	LFIsp &lsc_simd_u16(cv::Mat &img);
	LFIsp &lsc_simd_u8(cv::Mat &img);

	LFIsp &awb_simd_u16(cv::Mat &img);
	LFIsp &awb_simd_u8(cv::Mat &img);

	LFIsp &lsc_awb_simd_u16(cv::Mat &img);
	LFIsp &lsc_awb_simd_u8(cv::Mat &img);

	LFIsp &raw_to_8bit_with_gains_simd_u16(cv::Mat &dst_8u, float exposure);
	LFIsp &raw_to_8bit_with_gains_simd_u8(cv::Mat &dst_8u, float exposure);

	LFIsp &ccm_fixed_u16(cv::Mat &img);
	LFIsp &ccm_fixed_u8(cv::Mat &img);
};

#endif // LFISP_H

// #ifndef ISP_H
// #define ISP_H

// #include "lfparams.h"
// #include "utils.h"

// #include <algorithm>
// #include <immintrin.h>
// #include "json.hpp"
// #include <limits> // 必须包含，用于 numeric_limits
// #include <opencv2/core.hpp>
// #include <opencv2/core/hal/interface.h>
// #include <opencv2/core/types.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/opencv.hpp>
// #include <stdexcept>
// #include <type_traits>
// #include <vector>

// using json = nlohmann::json;

// template <typename T>
// class LFIsp {
// public:
// 	struct ResampleMaps {
// 		std::vector<cv::Mat> extract;
// 		std::vector<cv::Mat> dehex;
// 	} maps;

// public:
// 	explicit LFIsp() { cv::setNumThreads(cv::getNumberOfCPUs()); }
// 	explicit LFIsp(const LFParamsSource &config, const cv::Mat &lfp_img,
// 				   const cv::Mat &wht_img) {
// 		cv::setNumThreads(cv::getNumberOfCPUs());
// 		set_lf_img(lfp_img);
// 		set_white_img(wht_img);
// 		set_config(config);
// 	}
// 	explicit LFIsp(const json &json_config, const cv::Mat &lfp_img,
// 				   const cv::Mat &wht_img) {
// 		cv::setNumThreads(cv::getNumberOfCPUs());
// 		set_lf_img(lfp_img);
// 		set_white_img(wht_img);
// 		set_config(json_config);
// 	}

// 	LFParamsSource getConfig() const { return config_; }
// 	cv::Mat &getResult() { return lfp_img_; }
// 	const cv::Mat &getPreviewResult() const { return preview_img_; }
// 	const cv::Mat &getResult() const { return lfp_img_; }
// 	const std::vector<cv::Mat> &getSAIS() const { return sais; }

// 	LFParamsSource &get_config() { return config_; }

// 	LFIsp &set_config(const LFParamsSource &new_config) {
// 		config_ = new_config;
// 		prepare_ccm_fixed_point();

// 		return *this;
// 	}

// 	LFIsp &set_config(const json &json_settings) {
// 		LFParamsSource new_config; // 使用默认值初始化

// 		// 1. 解析 Bayer 格式字符串转枚举
// 		// filter_lfp_json 保证了只有 "BGGR" (Gen1) 或 "GRBG" (Gen2)
// 		// 等标准字符串
// 		if (json_settings.contains("bay")) {
// 			std::string bay_str = json_settings["bay"].get<std::string>();
// 			if (bay_str == "GRBG")
// 				new_config.bayer = BayerPattern::GRBG;
// 			else if (bay_str == "RGGB")
// 				new_config.bayer = BayerPattern::RGGB;
// 			else if (bay_str == "GBRG")
// 				new_config.bayer = BayerPattern::GBRG;
// 			else if (bay_str == "BGGR")
// 				new_config.bayer = BayerPattern::BGGR;
// 		}

// 		// 2. 位深
// 		if (json_settings.contains("bit")) {
// 			new_config.bitDepth = json_settings["bit"].get<int>();
// 		}

// 		// 3. 黑/白电平 (BLC)
// 		// JSON 中提供的是 4 通道的数组 (vector)，但 LFParamsSource
// 		// 目前用的是单一直流分量。 通常取第一个分量 (Gr) 或者做平均即可。对于
// 		// Lytro 相机，4个通道通常是一致的。
// 		if (json_settings.contains("blc")) {
// 			const auto &blc = json_settings["blc"];

// 			if (blc.contains("black") && blc["black"].is_array()
// 				&& !blc["black"].empty()) {
// 				// 取第 0 个元素的黑电平
// 				new_config.black_level = blc["black"][0].get<int>();
// 			}

// 			if (blc.contains("white") && blc["white"].is_array()
// 				&& !blc["white"].empty()) {
// 				// 取第 0 个元素的白电平 (饱和值)
// 				new_config.white_level = blc["white"][0].get<int>();
// 			}
// 		}

// 		// 4. AWB 增益
// 		// filter_lfp_json 已经将增益顺序调整为与 Bayer Pattern
// 		// 对应的顺序，直接赋值即可
// 		if (json_settings.contains("awb") && json_settings["awb"].is_array()) {
// 			new_config.awb_gains =
// 				json_settings["awb"].get<std::vector<float>>();
// 		}

// 		// 5. CCM 矩阵 (3x3 平铺为 9 float)
// 		if (json_settings.contains("ccm") && json_settings["ccm"].is_array()) {
// 			new_config.ccm_matrix =
// 				json_settings["ccm"].get<std::vector<float>>();
// 		}

// 		// 6. Gamma 值
// 		if (json_settings.contains("gam")) {
// 			new_config.gamma = json_settings["gam"].get<float>();
// 		}

// 		// 7. 曝光补偿 (可选)
// 		// 虽然 LFParamsSource 目前没有 exposure
// 		// 字段，如果将来需要数字增益补偿，可以在这里处理 float exp_bias =
// 		// json_settings.value("exp", 0.0f); new_config.digital_gain = pow(2.0f,
// 		// exp_bias);

// 		// 调用原本的 set_config 进行各种预计算 (如定点化 CCM)
// 		return set_config(new_config);
// 	}

// 	LFIsp &print_config() {
// 		std::cout << "\n================ [LFIsp Config] ================"
// 				  << std::endl;

// 		// 1. 基础信息
// 		std::cout << "Bayer Pattern : " << bayer_to_string(config_.bayer)
// 				  << std::endl;
// 		std::cout << "Bit Depth     : " << config_.bitDepth << "-bit"
// 				  << std::endl;
// 		std::cout << "Black Level   : " << config_.black_level << std::endl;
// 		std::cout << "White Level   : " << config_.white_level << std::endl;

// 		// 2. AWB Gains
// 		std::cout << "AWB Gains     : [ ";
// 		std::cout << std::fixed << std::setprecision(3); // 设置浮点精度
// 		if (config_.awb_gains.size() >= 4) {
// 			// 假设顺序是 Gr, R, B, Gb (或者根据你的定义打印对应的名字)
// 			std::cout << "Gr:" << config_.awb_gains[0] << ", ";
// 			std::cout << "R :" << config_.awb_gains[1] << ", ";
// 			std::cout << "B :" << config_.awb_gains[2] << ", ";
// 			std::cout << "Gb:" << config_.awb_gains[3];
// 		} else {
// 			for (float g : config_.awb_gains) std::cout << g << " ";
// 		}
// 		std::cout << " ]" << std::endl;

// 		// 3. Gamma
// 		std::cout << "Gamma         : " << config_.gamma << std::endl;

// 		// 4. CCM Matrix (格式化为 3x3 矩阵)
// 		std::cout << "CCM Matrix    :" << std::endl;
// 		if (config_.ccm_matrix.size() == 9) {
// 			for (int i = 0; i < 3; ++i) {
// 				std::cout << "                [ ";
// 				for (int j = 0; j < 3; ++j) {
// 					float val = config_.ccm_matrix[i * 3 + j];
// 					// setw(8) 保证对齐, showpos 显示正号
// 					std::cout << std::showpos << std::setw(8) << val << " ";
// 				}
// 				std::cout << std::noshowpos << "]" << std::endl;
// 			}
// 		} else {
// 			std::cout << "                (Invalid size: "
// 					  << config_.ccm_matrix.size() << ")" << std::endl;
// 		}

// 		std::cout << "================================================"
// 				  << std::endl;

// 		// 恢复默认 cout 格式，以免影响后续打印
// 		std::cout << std::defaultfloat;

// 		return *this;
// 	}

// 	LFIsp &updateImage(const cv::Mat &new_image, bool isWhite = false) {
// 		if (new_image.empty()) {
// 			throw std::runtime_error("Input image is empty.");
// 		}

// 		if (isWhite) {
// 			generate_lsc_maps(new_image);
// 		} else {
// 			lfp_img_ = new_image;
// 			// lfp_img_ = new_image.clone();
// 		}

// 		return *this;
// 	}

// 	LFIsp &set_lf_img(const cv::Mat &img) {
// 		if (img.empty()) {
// 			throw std::runtime_error("LF image is empty.");
// 		}

// 		lfp_img_ = img;

// 		return *this;
// 	}

// 	LFIsp &set_white_img(const cv::Mat &img) {
// 		if (img.empty()) {
// 			throw std::runtime_error("White image is empty.");
// 		}

// 		generate_lsc_maps(img);

// 		return *this;
// 	}

// 	LFIsp &preview(float exposure = 1.5f) {
// 		if (lfp_img_.empty())
// 			return *this;

// 		int rows = lfp_img_.rows;
// 		int cols = lfp_img_.cols;

// 		// 1. 准备中间缓冲 (8-bit Raw)
// 		// 这一步非常关键：我们在转 8-bit 的同时就把 AWB 和 LSC 做了
// 		cv::Mat raw_8u(rows, cols, CV_8UC1);

// 		// 2. 确保 LSC 表存在
// 		if (lsc_gain_map_int_.empty()) {
// 			// 简单的兜底
// 			if (!lsc_gain_map_.empty())
// 				lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0);
// 			else
// 				// 如果没有 LSC，传空指针，内核会自动处理
// 				lsc_gain_map_int_.release();
// 		}

// 		// 3. 执行 SIMD 核：Raw(u16) -> LSC -> AWB -> Raw(u8)
// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			raw_to_8bit_with_gains_simd(raw_8u, exposure);
// 		} else if constexpr (std::is_same_v<T, uint8_t>) {
// 			raw_to_8bit_with_gains_simd_u8(raw_8u, exposure);
// 		}

// 		// 4. 标准 Demosaic (8-bit)
// 		// OpenCV 处理 8-bit Bayer 非常快 (< 10ms @ 20MP)
// 		int code = get_demosaic_code(config_.bayer, false);
// 		cv::demosaicing(raw_8u, preview_img_, code);

// 		return *this;
// 	}

// 	LFIsp &blc() {
// 		// 1. 基础检查
// 		if (lfp_img_.empty())
// 			return *this;

// 		cv::subtract(lfp_img_, cv::Scalar(config_.black_level), lfp_img_);

// 		return *this;
// 	}

// 	LFIsp &dpc(int threshold = 100) {
// 		// 1. 基础检查
// 		if (lfp_img_.empty())
// 			return *this;

// 		int rows = lfp_img_.rows;
// 		int cols = lfp_img_.cols;

// 		// 忽略边缘 2 个像素 (因为 Bayer 模式同色邻居偏移量是 2)
// 		int border = 2;

// 		// 2. 最普通的双重循环 (串行执行)
// 		for (int r = border; r < rows - border; ++r) {
// 			// 获取行指针 (为了不至于慢到离谱，还是用指针访问，不用 .at())
// 			T *ptr_curr = lfp_img_.template ptr<T>(r);
// 			const T *ptr_up = lfp_img_.template ptr<T>(r - 2);
// 			const T *ptr_down = lfp_img_.template ptr<T>(r + 2);

// 			for (int c = border; c < cols - border; ++c) {
// 				T center = ptr_curr[c];

// 				// 3. 获取同色十字邻居 (上、下、左、右，步长为2)
// 				T val_L = ptr_curr[c - 2];
// 				T val_R = ptr_curr[c + 2];
// 				T val_U = ptr_up[c];
// 				T val_D = ptr_down[c];

// 				// 4. 找邻域内的最大值和最小值 (标量比较)
// 				T min_val = val_L;
// 				if (val_R < min_val)
// 					min_val = val_R;
// 				if (val_U < min_val)
// 					min_val = val_U;
// 				if (val_D < min_val)
// 					min_val = val_D;

// 				T max_val = val_L;
// 				if (val_R > max_val)
// 					max_val = val_R;
// 				if (val_U > max_val)
// 					max_val = val_U;
// 				if (val_D > max_val)
// 					max_val = val_D;

// 				// 5. 判定坏点 (转 int 防止溢出)
// 				// 必须比最大值还大 threshold，或比最小值还小 threshold
// 				bool is_hot = ((int)center > (int)max_val + threshold);
// 				bool is_dead = ((int)center < (int)min_val - threshold);

// 				// 6. 如果是坏点，执行修复
// 				if (is_hot || is_dead) {
// 					// 计算梯度 (Gradient)
// 					int grad_h = std::abs((int)val_L - (int)val_R);
// 					int grad_v = std::abs((int)val_U - (int)val_D);

// 					if (grad_h < grad_v) {
// 						// 水平梯度小，用左右平均
// 						ptr_curr[c] = static_cast<T>((val_L + val_R) / 2);
// 					} else if (grad_v < grad_h) {
// 						// 垂直梯度小，用上下平均
// 						ptr_curr[c] = static_cast<T>((val_U + val_D) / 2);
// 					} else {
// 						// 梯度一样，用四邻域平均
// 						ptr_curr[c] =
// 							static_cast<T>((val_L + val_R + val_U + val_D) / 4);
// 					}
// 				}
// 			}
// 		}

// 		return *this;
// 	}

// 	LFIsp &lsc() {
// 		// 1. 基础检查
// 		if (lfp_img_.empty() || lsc_gain_map_.empty())
// 			return *this;
// 		if (lfp_img_.size() != lsc_gain_map_.size())
// 			return *this;

// 		// 2. 类型转换 (u16 -> float)
// 		// 这一步非常慢，需要申请一张巨大的临时表 (20MP * 4B = 80MB)
// 		cv::Mat float_img;
// 		lfp_img_.convertTo(float_img, CV_32F);

// 		// 3. 执行乘法 (float * float)
// 		// 结果存回 float_img
// 		cv::multiply(float_img, lsc_gain_map_, float_img);

// 		// 4. 转回原类型 (float -> u16)
// 		// convertTo 内部会自动处理饱和截断 (Saturated Cast)
// 		// 例如 > 65535 会自动变 65535
// 		float_img.convertTo(lfp_img_, lfp_img_.type());

// 		return *this;
// 	}

// 	LFIsp &awb() {
// 		// 1. 基础检查
// 		if (lfp_img_.empty())
// 			return *this;

// 		int rows = lfp_img_.rows;
// 		int cols = lfp_img_.cols;

// 		// 获取 4 个增益标量
// 		float g_tl = config_.awb_gains[0];
// 		float g_tr = config_.awb_gains[1];
// 		float g_bl = config_.awb_gains[2];
// 		float g_br = config_.awb_gains[3];

// 		// T类型的最大值 (用于防溢出截断)
// 		const float max_val = static_cast<float>(std::numeric_limits<T>::max());

// 		// 2. 最普通的双重循环 (串行)
// 		for (int r = 0; r < rows; ++r) {
// 			T *ptr = lfp_img_.template ptr<T>(r);

// 			for (int c = 0; c < cols; ++c) {
// 				// 根据 (r, c) 的奇偶性决定使用哪个增益
// 				// 这就是最原始的教科书逻辑
// 				float gain;
// 				if (r % 2 == 0) {
// 					// 偶数行: TL, TR, TL, TR...
// 					gain = (c % 2 == 0) ? g_tl : g_tr;
// 				} else {
// 					// 奇数行: BL, BR, BL, BR...
// 					gain = (c % 2 == 0) ? g_bl : g_br;
// 				}

// 				// 计算：转float -> 乘法
// 				float val = static_cast<float>(ptr[c]) * gain;

// 				// 截断：防止溢出
// 				if (val > max_val) {
// 					val = max_val;
// 				}

// 				// 存回
// 				ptr[c] = static_cast<T>(val);
// 			}
// 		}

// 		return *this;
// 	}

// 	LFIsp &raw_process() {
// 		if (lfp_img_.empty())
// 			return *this;
// 		blc().dpc().lsc().awb();

// 		return *this;
// 	}

// 	LFIsp &demosaic() {
// 		// 1. 检查必要条件
// 		if (lfp_img_.empty())
// 			return *this;

// 		// 2. 防止重复 Demosaic
// 		if (lfp_img_.channels() != 1)
// 			return *this;

// 		int code = get_demosaic_code(config_.bayer, false);
// 		cv::demosaicing(lfp_img_, lfp_img_, code);

// 		return *this;
// 	}

// 	LFIsp &ccm() {
// 		// 1. 基础检查
// 		if (lfp_img_.empty())
// 			return *this;
// 		// CCM 必须在 Demosaic 之后，通常是 RGB 3通道
// 		if (lfp_img_.channels() != 3)
// 			return *this;

// 		// 2. 构造变换矩阵
// 		// 假设 config_.ccm_matrix 是 std::vector<float> (size=9)
// 		// 使用外部数据指针构造 cv::Mat，不产生内存拷贝，极快
// 		// (void*) 强转是为了兼容 OpenCV 的构造函数签名
// 		cv::Mat m(3, 3, CV_32F, (void *)config_.ccm_matrix.data());

// 		// 3. 执行变换
// 		cv::transform(lfp_img_, lfp_img_, m);

// 		return *this;
// 	}

// 	LFIsp &gc() { return *this; }

// 	// 1. BLC: 模板分发
// 	LFIsp &blc_fast() {
// 		if (lfp_img_.empty())
// 			return *this;

// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			return blc_simd_u16(lfp_img_);
// 		} else if constexpr (std::is_same_v<T, uint8_t>) {
// 			return blc_simd_u8(lfp_img_);
// 		} else {
// 			return *this;
// 		}
// 	}

// 	// 2. DPC: 模板分发 + 阈值适配
// 	LFIsp &dpc_fast(int threshold = 100) {
// 		if (lfp_img_.empty())
// 			return *this;

// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			// 针对 16-bit 通道 (u16)：
// 			return dpc_simd_u16(lfp_img_, threshold << (config_.bitDepth - 10));
// 		} else if constexpr (std::is_same_v<T, uint8_t>) {
// 			// 针对 8-bit 通道 (u8)：
// 			return dpc_simd_u8(lfp_img_, std::max(1, threshold >> 2));
// 		} else {
// 			return *this;
// 		}
// 	}

// 	// 3. LSC: 模板分发 + 懒加载 Map
// 	LFIsp &lsc_fast() {
// 		if (lfp_img_.empty()) {
// 			return *this;
// 		}

// 		// 检查 LSC Map 是否准备好
// 		if (lsc_gain_map_int_.empty()
// 			|| lsc_gain_map_int_.size() != lfp_img_.size()) {
// 			if (lsc_gain_map_.empty()) {
// 				return *this;
// 			}

// 			lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0f);
// 		}

// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			return lsc_simd_u16(lfp_img_);
// 		} else if constexpr (std::is_same_v<T, uint8_t>) {
// 			return lsc_simd_u8(lfp_img_);
// 		} else {
// 			return *this;
// 		}
// 	}

// 	// 4. AWB: 模板分发
// 	LFIsp &awb_fast() {
// 		if (lfp_img_.empty())
// 			return *this;

// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			return awb_simd_u16(lfp_img_);
// 		} else if constexpr (std::is_same_v<T, uint8_t>) {
// 			return awb_simd_u8(lfp_img_);
// 		} else {
// 			return *this;
// 		}
// 	}

// 	LFIsp &lsc_awb_fused_fast() {
// 		if (lfp_img_.empty())
// 			return *this;

// 		// 1. 兜底：确保定点 LSC 表存在
// 		if (lsc_gain_map_int_.empty()
// 			|| lsc_gain_map_int_.size() != lfp_img_.size()) {
// 			if (lsc_gain_map_.empty()) {
// 				// 极端的兜底：生成全 1.0
// 				lsc_gain_map_ = cv::Mat::ones(lfp_img_.size(), CV_32F);
// 			}
// 			// 4096.0 对应 Q12 定点数
// 			lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0);
// 		}

// 		// 2. 统一分发 (Consistent Dispatching)
// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			return lsc_awb_simd_u16(lfp_img_);
// 		} else if (std::is_same_v<T, uint8_t>) {
// 			return lsc_awb_simd_u8(lfp_img_);
// 		} else {
// 			return *this;
// 		}
// 	}

// 	LFIsp &raw_process_fast() {
// 		if (lfp_img_.empty())
// 			return *this;

// 		blc_fast().dpc_fast().lsc_fast().awb_fast();
// 		// blc_fast().dpc_fast().lsc_awb_fused_fast();

// 		return *this;
// 	}

// 	LFIsp &ccm_fast() {
// 		if (lfp_img_.empty())
// 			return *this;
// 		if (lfp_img_.channels() != 3)
// 			return *this;

// 		// 兜底：如果定点矩阵没准备好
// 		if (ccm_matrix_int_.empty())
// 			prepare_ccm_fixed_point();
// 		int depth = lfp_img_.depth();
// 		if constexpr (std::is_same_v<T, uint16_t>) {
// 			return ccm_fixed_u16(lfp_img_);
// 		} else if (std::is_same_v<T, uint8_t>) {
// 			return ccm_fixed_u8(lfp_img_);
// 		} else {
// 			return *this;
// 		}
// 	}

// 	LFIsp &resample(bool dehex) {
// 		int num_views = maps.extract.size() / 2;
// 		cv::Mat src = preview_img_.empty() ? lfp_img_ : preview_img_;

// 		sais.clear();
// 		sais.resize(num_views);

// #pragma omp parallel for schedule(dynamic)
// 		for (int i = 0; i < num_views; ++i) {
// 			cv::Mat temp;
// 			cv::remap(src, temp, maps.extract[i * 2], maps.extract[i * 2 + 1],
// 					  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
// 			if (dehex) {
// 				cv::remap(temp, temp, maps.dehex[0], maps.dehex[1],
// 						  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
// 			}

// 			sais[i] = temp;
// 		}

// 		return *this;
// 	}

// 	LFIsp &color_equalize() {
// 		if (sais.empty())
// 			return *this;

// 		// 1. 找到中心视点 (Reference)
// 		int num_views = sais.size();
// 		int side_len = static_cast<int>(std::sqrt(num_views));
// 		int center_idx = (side_len / 2) * side_len + (side_len / 2);

// 		// 边界检查
// 		if (center_idx >= num_views)
// 			center_idx = 0;

// 		const cv::Mat &ref_img = sais[center_idx];

// 		// 2. [优化] 预计算参考图(中心视点)的统计信息
// 		// 避免在循环中重复计算 80 次，节省时间
// 		cv::Scalar ref_mean, ref_std;
// 		compute_lab_stats(ref_img, ref_mean, ref_std);

// // 3. 并行处理所有视点 (Source)
// #pragma omp parallel for schedule(dynamic)
// 		for (int i = 0; i < num_views; ++i) {
// 			if (i == center_idx)
// 				continue; // 跳过中心视点自己

// 			// 对当前视点应用 Reinhard 变换
// 			// sais[i] 会被原地修改
// 			apply_reinhard_transfer(sais[i], ref_mean, ref_std);
// 		}

// 		return *this;
// 	}

// private:
// 	LFParamsSource config_;

// 	cv::Mat preview_img_;
// 	cv::Mat lfp_img_, wht_img_;
// 	cv::Mat lsc_gain_map_, lsc_gain_map_int_;
// 	cv::Mat fused_gain_map_int_;
// 	std::vector<int32_t> ccm_matrix_int_;
// 	std::vector<cv::Mat> sais;

// 	std::string bayer_to_string(BayerPattern p) const {
// 		switch (p) {
// 			case BayerPattern::GRBG:
// 				return "GRBG";
// 			case BayerPattern::RGGB:
// 				return "RGGB";
// 			case BayerPattern::GBRG:
// 				return "GBRG";
// 			case BayerPattern::BGGR:
// 				return "BGGR";
// 			default:
// 				return "Unknown";
// 		}
// 	}

// 	LFIsp &blc_simd_u16(cv::Mat &img) {
// 		uint16_t bl_val = static_cast<uint16_t>(config_.black_level);
// 		__m256i v_bl = _mm256_set1_epi16(bl_val);

// 		int rows = img.rows;
// 		int cols = img.cols;
// 		if (img.isContinuous()) {
// 			cols = rows * cols;
// 			rows = 1;
// 		}

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint16_t *ptr = img.ptr<uint16_t>(r);
// 			int c = 0;

// 			for (; c <= cols - 16; c += 16) {
// 				__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr + c));
// 				v_src = _mm256_subs_epu16(v_src, v_bl);
// 				_mm256_storeu_si256((__m256i *)(ptr + c), v_src);
// 			}

// 			for (; c < cols; ++c) {
// 				int val = ptr[c];
// 				ptr[c] = (val > bl_val) ? (val - bl_val) : 0;
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &blc_simd_u8(cv::Mat &img) {
// 		uint8_t bl_val = static_cast<uint8_t>(config_.black_level);
// 		__m256i v_bl = _mm256_set1_epi8(bl_val);

// 		int rows = img.rows;
// 		int cols = img.cols;
// 		if (img.isContinuous()) {
// 			cols = rows * cols;
// 			rows = 1;
// 		}

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint8_t *ptr = img.ptr<uint8_t>(r);
// 			int c = 0;

// 			for (; c <= cols - 32; c += 32) {
// 				__m256i v_src = _mm256_loadu_si256((const __m256i *)(ptr + c));

// 				// 8-bit 饱和减法
// 				v_src = _mm256_subs_epu8(v_src, v_bl);

// 				_mm256_storeu_si256((__m256i *)(ptr + c), v_src);
// 			}

// 			// 处理尾部
// 			for (; c < cols; ++c) {
// 				int val = ptr[c];
// 				ptr[c] = (val > bl_val) ? (val - bl_val) : 0;
// 			}
// 		}
// 		return *this;
// 	}

// 	inline void dpc_scalar_kernel(int r, int c, T *ptr_curr, const T *ptr_up,
// 								  const T *ptr_down, int threshold) {
// 		T center = ptr_curr[c];
// 		T val_L = ptr_curr[c - 2];
// 		T val_R = ptr_curr[c + 2];
// 		T val_U = ptr_up[c];
// 		T val_D = ptr_down[c];

// 		// 极值 (利用 initializer_list 简化代码)
// 		T min_val = std::min({val_L, val_R, val_U, val_D});
// 		T max_val = std::max({val_L, val_R, val_U, val_D});

// 		// 坏点判定
// 		bool is_hot = (center > max_val) && (center - max_val > threshold);
// 		bool is_dead = (center < min_val) && (min_val - center > threshold);

// 		if (is_hot || is_dead) {
// 			// 梯度
// 			int grad_h = std::abs((int)val_L - (int)val_R);
// 			int grad_v = std::abs((int)val_U - (int)val_D);

// 			if (grad_h < grad_v)
// 				ptr_curr[c] = (val_L + val_R) / 2;
// 			else if (grad_v < grad_h)
// 				ptr_curr[c] = (val_U + val_D) / 2;
// 			else
// 				ptr_curr[c] = (val_L + val_R + val_U + val_D) / 4;
// 		}
// 	}

// 	LFIsp &dpc_simd_u16(cv::Mat &img, int threshold) {
// 		int rows = img.rows;
// 		int cols = img.cols;
// 		int border = 2; // 必须 >= 2

// 		// 准备常量
// 		// 符号位翻转掩码 (用于无符号比较技巧)
// 		__m256i v_sign_bit = _mm256_set1_epi16((short)0x8000);
// 		// 阈值向量
// 		__m256i v_thresh = _mm256_set1_epi16((short)threshold);

// #pragma omp parallel for
// 		for (int r = border; r < rows - border; ++r) {
// 			uint16_t *ptr_curr = img.template ptr<uint16_t>(r);
// 			const uint16_t *ptr_up = img.template ptr<uint16_t>(r - 2);
// 			const uint16_t *ptr_down = img.template ptr<uint16_t>(r + 2);

// 			int c = border;
// 			// 每次处理 16 个像素
// 			for (; c <= cols - border - 16; c += 16) {
// 				// 1. 加载数据 (Load Unaligned)
// 				// 中心点
// 				__m256i v_curr =
// 					_mm256_loadu_si256((const __m256i *)(ptr_curr + c));
// 				// 十字邻域 (Offset +/- 2)
// 				__m256i v_L =
// 					_mm256_loadu_si256((const __m256i *)(ptr_curr + c - 2));
// 				__m256i v_R =
// 					_mm256_loadu_si256((const __m256i *)(ptr_curr + c + 2));
// 				__m256i v_U = _mm256_loadu_si256((const __m256i *)(ptr_up + c));
// 				__m256i v_D =
// 					_mm256_loadu_si256((const __m256i *)(ptr_down + c));

// 				// 2. 极值检测 (Min/Max)
// 				__m256i v_min = _mm256_min_epu16(v_L, v_R);
// 				v_min = _mm256_min_epu16(v_min, v_U);
// 				v_min = _mm256_min_epu16(v_min, v_D);

// 				__m256i v_max = _mm256_max_epu16(v_L, v_R);
// 				v_max = _mm256_max_epu16(v_max, v_U);
// 				v_max = _mm256_max_epu16(v_max, v_D);

// 				// 3. 坏点判定 (Bad Pixel Detection)
// 				// Hot: curr > max + th  --> (curr - th) > max (使用饱和减法)
// 				// Dead: curr < min - th --> (curr + th) < min (使用饱和加法)

// 				// 饱和运算 (Saturated Arithmetic) 避免溢出
// 				__m256i v_curr_minus_th = _mm256_subs_epu16(v_curr, v_thresh);
// 				__m256i v_curr_plus_th = _mm256_adds_epu16(v_curr, v_thresh);

// 				// 无符号比较技巧: (A ^ 0x8000) > (B ^ 0x8000) 等价于 unsigned A
// 				// > B
// 				__m256i v_cmp_hot_lhs =
// 					_mm256_xor_si256(v_curr_minus_th, v_sign_bit);
// 				__m256i v_cmp_hot_rhs = _mm256_xor_si256(v_max, v_sign_bit);
// 				__m256i mask_hot =
// 					_mm256_cmpgt_epi16(v_cmp_hot_lhs, v_cmp_hot_rhs);

// 				__m256i v_cmp_dead_lhs = _mm256_xor_si256(v_min, v_sign_bit);
// 				__m256i v_cmp_dead_rhs =
// 					_mm256_xor_si256(v_curr_plus_th, v_sign_bit);
// 				__m256i mask_dead =
// 					_mm256_cmpgt_epi16(v_cmp_dead_lhs, v_cmp_dead_rhs);

// 				// 合并坏点掩膜 (Bad = Hot | Dead)
// 				__m256i mask_bad = _mm256_or_si256(mask_hot, mask_dead);

// 				// 优化：如果没有坏点，直接跳过计算 (大大提升 Clean Image
// 				// 的性能)
// 				if (_mm256_testz_si256(mask_bad, mask_bad)) {
// 					continue;
// 				}

// 				// 4. 梯度计算 (Gradient)
// 				// abs_diff = max - min (unsigned)
// 				__m256i grad_h = _mm256_subs_epu16(_mm256_max_epu16(v_L, v_R),
// 												   _mm256_min_epu16(v_L, v_R));
// 				__m256i grad_v = _mm256_subs_epu16(_mm256_max_epu16(v_U, v_D),
// 												   _mm256_min_epu16(v_U, v_D));

// 				// 5. 计算修复值 (Fix Value)
// 				// avg = (A + B + 1) >> 1
// 				__m256i fix_h = _mm256_avg_epu16(v_L, v_R);
// 				__m256i fix_v = _mm256_avg_epu16(v_U, v_D);

// 				// 四邻域平均: (fix_h + fix_v + 1) >> 1
// 				__m256i fix_all = _mm256_avg_epu16(fix_h, fix_v);

// 				// 6. 梯度选择 (Gradient Selection)
// 				// if (grad_h < grad_v) use fix_h
// 				// if (grad_v < grad_h) use fix_v
// 				// else use fix_all

// 				// 构造掩膜
// 				// grad_h < grad_v  <==>  grad_v > grad_h
// 				__m256i v_gh_sign = _mm256_xor_si256(grad_h, v_sign_bit);
// 				__m256i v_gv_sign = _mm256_xor_si256(grad_v, v_sign_bit);

// 				__m256i mask_use_h = _mm256_cmpgt_epi16(
// 					v_gv_sign, v_gh_sign); // V > H -> H is better
// 				__m256i mask_use_v = _mm256_cmpgt_epi16(
// 					v_gh_sign, v_gv_sign); // H > V -> V is better

// 				// 混合修复值
// 				// 先默认用 fix_all
// 				__m256i v_fixed = fix_all;
// 				// 如果 H 更好，用 H 覆盖
// 				v_fixed = _mm256_blendv_epi8(v_fixed, fix_h, mask_use_h);
// 				// 如果 V 更好，用 V 覆盖 (注意:
// 				// H和V不可能同时更好，所以顺序无所谓)
// 				v_fixed = _mm256_blendv_epi8(v_fixed, fix_v, mask_use_v);

// 				// 7. 最终混合 (Original vs Fixed)
// 				// 只在 mask_bad 为 1 的地方写入 v_fixed，其他地方保持 v_curr
// 				__m256i v_result =
// 					_mm256_blendv_epi8(v_curr, v_fixed, mask_bad);

// 				// 8. 存回
// 				_mm256_storeu_si256((__m256i *)(ptr_curr + c), v_result);
// 			}

// 			// 处理尾部 (Scalar Fallback)
// 			for (; c < cols - border; ++c) {
// 				dpc_scalar_kernel(r, c, ptr_curr, ptr_up, ptr_down, threshold);
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &dpc_simd_u8(cv::Mat &img, int threshold) {
// 		int rows = img.rows;
// 		int cols = img.cols;
// 		int border = 2;

// 		// 准备常量 (扩展到 16-bit)
// 		__m256i v_sign_bit = _mm256_set1_epi16((short)0x8000);
// 		__m256i v_thresh = _mm256_set1_epi16((short)threshold);

// #pragma omp parallel for
// 		for (int r = border; r < rows - border; ++r) {
// 			uint8_t *ptr_curr = img.template ptr<uint8_t>(r);
// 			const uint8_t *ptr_up = img.template ptr<uint8_t>(r - 2);
// 			const uint8_t *ptr_down = img.template ptr<uint8_t>(r + 2);

// 			int c = border;
// 			// 每次处理 16 个像素 (因为要扩展成 u16，占满 256-bit 寄存器)
// 			for (; c <= cols - border - 16; c += 16) {
// 				// 1. 加载数据 (Load 16x u8 -> 128-bit)
// 				__m128i v_curr_8 =
// 					_mm_loadu_si128((const __m128i *)(ptr_curr + c));
// 				__m128i v_L_8 =
// 					_mm_loadu_si128((const __m128i *)(ptr_curr + c - 2));
// 				__m128i v_R_8 =
// 					_mm_loadu_si128((const __m128i *)(ptr_curr + c + 2));
// 				__m128i v_U_8 = _mm_loadu_si128((const __m128i *)(ptr_up + c));
// 				__m128i v_D_8 =
// 					_mm_loadu_si128((const __m128i *)(ptr_down + c));

// 				// 2. 扩展为 u16 (Expand to 256-bit)
// 				// 只有扩展后，才能安全地做加法和梯度计算而不溢出
// 				__m256i v_curr = _mm256_cvtepu8_epi16(v_curr_8);
// 				__m256i v_L = _mm256_cvtepu8_epi16(v_L_8);
// 				__m256i v_R = _mm256_cvtepu8_epi16(v_R_8);
// 				__m256i v_U = _mm256_cvtepu8_epi16(v_U_8);
// 				__m256i v_D = _mm256_cvtepu8_epi16(v_D_8);

// 				// --- 以下逻辑与 dpc_simd_u16 完全一致 ---

// 				// 3. 极值检测
// 				__m256i v_min = _mm256_min_epu16(_mm256_min_epu16(v_L, v_R),
// 												 _mm256_min_epu16(v_U, v_D));
// 				__m256i v_max = _mm256_max_epu16(_mm256_max_epu16(v_L, v_R),
// 												 _mm256_max_epu16(v_U, v_D));

// 				// 4. 坏点判定
// 				// Hot: (curr - th) > max
// 				__m256i v_hot = _mm256_cmpgt_epi16(
// 					_mm256_xor_si256(_mm256_subs_epu16(v_curr, v_thresh),
// 									 v_sign_bit),
// 					_mm256_xor_si256(v_max, v_sign_bit));
// 				// Dead: (curr + th) < min
// 				__m256i v_dead = _mm256_cmpgt_epi16(
// 					_mm256_xor_si256(v_min, v_sign_bit),
// 					_mm256_xor_si256(_mm256_adds_epu16(v_curr, v_thresh),
// 									 v_sign_bit));
// 				__m256i mask_bad = _mm256_or_si256(v_hot, v_dead);

// 				// 5. 快速跳过 (Optimization)
// 				if (_mm256_testz_si256(mask_bad, mask_bad)) {
// 					continue;
// 				}

// 				// 6. 梯度与修复
// 				__m256i g_h = _mm256_subs_epu16(_mm256_max_epu16(v_L, v_R),
// 												_mm256_min_epu16(v_L, v_R));
// 				__m256i g_v = _mm256_subs_epu16(_mm256_max_epu16(v_U, v_D),
// 												_mm256_min_epu16(v_U, v_D));

// 				__m256i fix_h = _mm256_avg_epu16(v_L, v_R);
// 				__m256i fix_v = _mm256_avg_epu16(v_U, v_D);
// 				__m256i fix_all = _mm256_avg_epu16(fix_h, fix_v);

// 				__m256i use_h =
// 					_mm256_cmpgt_epi16(_mm256_xor_si256(g_v, v_sign_bit),
// 									   _mm256_xor_si256(g_h, v_sign_bit));
// 				__m256i use_v =
// 					_mm256_cmpgt_epi16(_mm256_xor_si256(g_h, v_sign_bit),
// 									   _mm256_xor_si256(g_v, v_sign_bit));

// 				__m256i v_fixed = fix_all;
// 				v_fixed = _mm256_blendv_epi8(v_fixed, fix_h, use_h);
// 				v_fixed = _mm256_blendv_epi8(v_fixed, fix_v, use_v);

// 				// 混合结果 (Result in u16)
// 				__m256i v_res_16 =
// 					_mm256_blendv_epi8(v_curr, v_fixed, mask_bad);

// 				// 7. Pack 回 u8
// 				// packus 将 256-bit (16x u16) 压缩为 128-bit (16x u8)
// 				// 注意：packus 实际上输出
// 				// 256-bit，高128位是重复或无效的，我们需要提取低128位
// 				// 这里需要一点技巧：_mm256_packus_epi16 输入两个
// 				// 256bit，输出一个 256bit 我们只有一个 256bit 数据
// 				// (v_res_16)，我们把它自己跟自己 pack，或者跟 0 pack

// 				// 更简单的做法：使用 128-bit 的 pack 指令，但这需要把 256bit
// 				// 拆开 推荐：使用 _mm256_packus_epi16，配合 Permute
// 				__m256i v_packed = _mm256_packus_epi16(v_res_16, v_res_16);

// 				// Pack 后的顺序是乱的 [A_lo, A_hi, B_lo, B_hi]，需要 Permute
// 				// 理顺
// 				v_packed =
// 					_mm256_permute4x64_epi64(v_packed, _MM_SHUFFLE(3, 1, 2, 0));

// 				// 现在 v_packed 的低 128 位就是我们要的 16 个 u8
// 				__m128i v_final = _mm256_castsi256_si128(v_packed);

// 				// 8. 存回
// 				_mm_storeu_si128((__m128i *)(ptr_curr + c), v_final);
// 			}

// 			// 处理尾部 (Scalar)
// 			for (; c < cols - border; ++c) {
// 				dpc_scalar_kernel(r, c, ptr_curr, ptr_up, ptr_down, threshold);
// 			}
// 		}
// 		return *this;
// 	}

// 	// 核心函数：处理白板 -> 分通道模糊 -> 计算增益 -> 生成定点数表
// 	void generate_lsc_maps(const cv::Mat &raw_wht) {
// 		int rows = raw_wht.rows;
// 		int cols = raw_wht.cols;

// 		// 1. 预处理：转浮点 + 减黑电平
// 		cv::Mat float_wht;
// 		raw_wht.convertTo(float_wht, CV_32F);

// 		float bl = static_cast<float>(config_.black_level);
// 		cv::subtract(float_wht, cv::Scalar(bl), float_wht);
// 		cv::max(float_wht, 1.0f, float_wht); // 兜底防止除0

// 		// 2. 拆分 Bayer 通道 (Split)
// 		// 目的：为了独立进行高斯模糊，不破坏 Bayer 结构
// 		int half_h = rows / 2;
// 		int half_w = cols / 2;
// 		std::vector<cv::Mat> channels(4);
// 		for (int k = 0; k < 4; ++k) channels[k].create(half_h, half_w, CV_32F);

// #pragma omp parallel for
// 		for (int r = 0; r < half_h; ++r) {
// 			const float *ptr_row0 = float_wht.ptr<float>(2 * r);
// 			const float *ptr_row1 = float_wht.ptr<float>(2 * r + 1);

// 			float *p0 = channels[0].ptr<float>(r);
// 			float *p1 = channels[1].ptr<float>(r);
// 			float *p2 = channels[2].ptr<float>(r);
// 			float *p3 = channels[3].ptr<float>(r);

// 			for (int c = 0; c < half_w; ++c) {
// 				p0[c] = ptr_row0[2 * c];	 // TL
// 				p1[c] = ptr_row0[2 * c + 1]; // TR
// 				p2[c] = ptr_row1[2 * c];	 // BL
// 				p3[c] = ptr_row1[2 * c + 1]; // BR
// 			}
// 		}

// // 3. 并行高斯模糊 (Blur)
// // 滤除 Sensor 噪声，保留 Lens Shading 趋势
// #pragma omp parallel for
// 		for (int k = 0; k < 4; ++k) {
// 			cv::GaussianBlur(channels[k], channels[k], cv::Size(7, 7), 0);
// 		}

// 		// 4. 【修改】寻找每个通道独立的局部最大值
// 		// 不再计算 Global Max，而是计算 Per-channel Max
// 		std::vector<double> maxVals(4);
// 		for (int k = 0; k < 4; ++k) {
// 			double localMax;
// 			cv::minMaxLoc(channels[k], nullptr, &localMax);
// 			if (localMax < 1e-6)
// 				localMax = 1.0; // 防除0
// 			maxVals[k] = localMax;
// 		}

// 		// 5. 【修改】计算增益 (使用各自通道的 Max)
// 		lsc_gain_map_.create(rows, cols, CV_32F);

// #pragma omp parallel for
// 		for (int r = 0; r < half_h; ++r) {
// 			float *dst0 = lsc_gain_map_.ptr<float>(2 * r);
// 			float *dst1 = lsc_gain_map_.ptr<float>(2 * r + 1);

// 			const float *p0 = channels[0].ptr<float>(r);
// 			const float *p1 = channels[1].ptr<float>(r);
// 			const float *p2 = channels[2].ptr<float>(r);
// 			const float *p3 = channels[3].ptr<float>(r);

// 			for (int c = 0; c < half_w; ++c) {
// 				// Gain_K = Max_K / Pixel_K
// 				// 这样无论白板本身偏什么色，中心点的增益永远是 1.0
// 				dst0[2 * c] = (float)maxVals[0] / p0[c];	 // TL
// 				dst0[2 * c + 1] = (float)maxVals[1] / p1[c]; // TR
// 				dst1[2 * c] = (float)maxVals[2] / p2[c];	 // BL
// 				dst1[2 * c + 1] = (float)maxVals[3] / p3[c]; // BR
// 			}
// 		}

// 		// 6. 生成定点数 Map (Float -> Uint16)
// 		// 放大 4096 倍 (Q12)，供 lsc_fast 中的 SIMD 使用
// 		lsc_gain_map_.convertTo(lsc_gain_map_int_, CV_16U, 4096.0);
// 	}

// 	LFIsp &lsc_simd_u16(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint16_t *ptr_src = img.template ptr<uint16_t>(r);
// 			const uint16_t *ptr_gain = lsc_gain_map_int_.ptr<uint16_t>(r);

// 			int c = 0;
// 			// 每次处理 16 个像素
// 			for (; c <= cols - 16; c += 16) {
// 				// Load Pixel (u16)
// 				__m256i v_src =
// 					_mm256_loadu_si256((const __m256i *)(ptr_src + c));
// 				// Load Gain (u16)
// 				__m256i v_gain =
// 					_mm256_loadu_si256((const __m256i *)(ptr_gain + c));

// 				// Unpack Low 8 -> 32-bit int
// 				__m256i v_src_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
// 				__m256i v_gain_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));
// 				// Mul & Shift ( >> 12 )
// 				__m256i v_res_lo = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);

// 				// Unpack High 8 -> 32-bit int
// 				__m256i v_src_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));
// 				__m256i v_gain_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));
// 				// Mul & Shift ( >> 12 )
// 				__m256i v_res_hi = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

// 				// Pack & Saturate (u32 -> u16)
// 				__m256i v_res = _mm256_packus_epi32(v_res_lo, v_res_hi);
// 				// 修正 lane 顺序
// 				v_res =
// 					_mm256_permute4x64_epi64(v_res, _MM_SHUFFLE(3, 1, 2, 0));

// 				_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res);
// 			}

// 			// Tail
// 			for (; c < cols; ++c) {
// 				uint32_t val = (uint32_t)ptr_src[c] * ptr_gain[c];
// 				val >>= 12;
// 				if (val > 65535)
// 					val = 65535;
// 				ptr_src[c] = (uint16_t)val;
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &lsc_simd_u8(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint8_t *ptr_src = img.template ptr<uint8_t>(r);
// 			const uint16_t *ptr_gain = lsc_gain_map_int_.ptr<uint16_t>(r);

// 			int c = 0;
// 			// 每次处理 16 个像素 (为了配合 Gain 的 u16 对齐)
// 			// 这里的瓶颈是 Gain 是 16位的，所以一次只能加载 16 个 gain
// 			for (; c <= cols - 16; c += 16) {
// 				// Load 16 Pixel (u8) -> 加载到 128位 寄存器
// 				__m128i v_src_small =
// 					_mm_loadu_si128((const __m128i *)(ptr_src + c));
// 				// Load 16 Gain (u16) -> 加载到 256位 寄存器
// 				__m256i v_gain =
// 					_mm256_loadu_si256((const __m256i *)(ptr_gain + c));

// 				// Unpack u8 -> u32 (Low 8)
// 				__m256i v_src_lo = _mm256_cvtepu8_epi32(v_src_small);
// 				// Unpack Gain u16 -> u32 (Low 8)
// 				__m256i v_gain_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));
// 				// Mul & Shift
// 				__m256i v_res_lo = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);

// 				// Unpack u8 -> u32 (High 8)
// 				// 先把高位移到低位
// 				__m128i v_src_hi_small =
// 					_mm_unpackhi_epi64(v_src_small, v_src_small);
// 				__m256i v_src_hi = _mm256_cvtepu8_epi32(v_src_hi_small);
// 				// Unpack Gain u16 -> u32 (High 8)
// 				__m256i v_gain_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));
// 				// Mul & Shift
// 				__m256i v_res_hi = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

// 				// Pack: u32 -> u16 -> u8
// 				// 1. Pack u32 -> u16 (带饱和)
// 				__m256i v_packed_16 = _mm256_packus_epi32(v_res_lo, v_res_hi);
// 				v_packed_16 = _mm256_permute4x64_epi64(v_packed_16,
// 													   _MM_SHUFFLE(3, 1, 2, 0));

// 				// 2. Pack u16 -> u8 (需要把 256bit 拆成两个 128bit 再 pack)
// 				__m128i v_packed_u8 =
// 					_mm_packus_epi16(_mm256_castsi256_si128(v_packed_16),
// 									 _mm256_extracti128_si256(v_packed_16, 1));

// 				// Store (16 bytes)
// 				_mm_storeu_si128((__m128i *)(ptr_src + c), v_packed_u8);
// 			}

// 			// Tail
// 			for (; c < cols; ++c) {
// 				uint32_t val = (uint32_t)ptr_src[c] * ptr_gain[c];
// 				val >>= 12;
// 				if (val > 255)
// 					val = 255;
// 				ptr_src[c] = (uint8_t)val;
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &awb_simd_u16(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// 		// 1. 准备定点数增益 (Q12格式: 1.0 = 4096)
// 		// 使用 4096 (2^12) 作为缩放因子，保证精度且防止 overflow
// 		// 假设 config_.awb_gains 顺序是 [TL, TR, BL, BR]
// 		const float scale = 4096.0f;
// 		uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
// 		uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
// 		uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
// 		uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

// 		// 2. 构造 AVX2 增益向量 (直接存寄存器，零内存访问)

// 		// Row 0 Pattern: TL, TR, TL, TR...
// 		// 构造 32位 组合: (g_tr << 16) | g_tl
// 		uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
// 		__m256i v_gain_row0 = _mm256_set1_epi32(p_row0);

// 		// Row 1 Pattern: BL, BR, BL, BR...
// 		uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
// 		__m256i v_gain_row1 = _mm256_set1_epi32(p_row1);

// #pragma omp parallel for
// 		for (int r = 0; r < rows; r += 2) {
// 			if (r + 1 >= rows)
// 				continue;

// 			uint16_t *ptr0 = img.template ptr<uint16_t>(r);
// 			uint16_t *ptr1 = img.template ptr<uint16_t>(r + 1);

// 			int c = 0;
// 			// 每次处理 16 个像素
// 			for (; c <= cols - 16; c += 16) {
// 				// [Row 0 Processing]
// 				__m256i v0 = _mm256_loadu_si256((const __m256i *)(ptr0 + c));

// 				// Unpack to 32-bit (Low & High)
// 				__m256i v0_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v0));
// 				__m256i v0_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v0, 1));

// 				// Unpack Gain (Low & High)
// 				// 注意：gain vector 已经是 packed u16，也需要 unpack 才能和
// 				// pixel 做 32位乘法
// 				__m256i vg0_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain_row0));
// 				__m256i vg0_hi = _mm256_cvtepu16_epi32(
// 					_mm256_extracti128_si256(v_gain_row0, 1));

// 				// Multiply & Shift Right 12
// 				v0_lo =
// 					_mm256_srli_epi32(_mm256_mullo_epi32(v0_lo, vg0_lo), 12);
// 				v0_hi =
// 					_mm256_srli_epi32(_mm256_mullo_epi32(v0_hi, vg0_hi), 12);

// 				// Pack & Saturate (Back to u16)
// 				v0 = _mm256_packus_epi32(v0_lo, v0_hi);
// 				v0 = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
// 				_mm256_storeu_si256((__m256i *)(ptr0 + c), v0);

// 				// [Row 1 Processing] (Logic is identical, just using ptr1 and
// 				// v_gain_row1)
// 				__m256i v1 = _mm256_loadu_si256((const __m256i *)(ptr1 + c));

// 				__m256i v1_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v1));
// 				__m256i v1_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v1, 1));

// 				__m256i vg1_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain_row1));
// 				__m256i vg1_hi = _mm256_cvtepu16_epi32(
// 					_mm256_extracti128_si256(v_gain_row1, 1));

// 				v1_lo =
// 					_mm256_srli_epi32(_mm256_mullo_epi32(v1_lo, vg1_lo), 12);
// 				v1_hi =
// 					_mm256_srli_epi32(_mm256_mullo_epi32(v1_hi, vg1_hi), 12);

// 				v1 = _mm256_packus_epi32(v1_lo, v1_hi);
// 				v1 = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
// 				_mm256_storeu_si256((__m256i *)(ptr1 + c), v1);
// 			}

// 			// 尾部处理 (Scalar Fallback)
// 			for (; c < cols; ++c) {
// 				// Row 0
// 				uint32_t val0 = (uint32_t)ptr0[c] * ((c % 2) ? g_tr : g_tl);
// 				val0 >>= 12;
// 				ptr0[c] = (val0 > 65535) ? 65535 : (uint16_t)val0;

// 				// Row 1
// 				uint32_t val1 = (uint32_t)ptr1[c] * ((c % 2) ? g_br : g_bl);
// 				val1 >>= 12;
// 				ptr1[c] = (val1 > 65535) ? 65535 : (uint16_t)val1;
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &awb_simd_u8(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// 		const float scale = 4096.0f; // Q12 定点数
// 		uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
// 		uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
// 		uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
// 		uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

// 		// 构造 u16 广播向量 (注意：u8 扩展后变成 u16，所以这里用 u16 匹配)
// 		// Row 0 Pattern: TL, TR, TL, TR... (packed as 16-bit integers)
// 		// _mm256_set1_epi32 会把 32位 复制8次。
// 		// 我们需要构造一个 32位整数包含 [TR, TL] (Little Endian: TL在低位)
// 		uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
// 		__m256i v_gain_row0 = _mm256_set1_epi32(p_row0);

// 		// Row 1 Pattern: BL, BR, BL, BR...
// 		uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
// 		__m256i v_gain_row1 = _mm256_set1_epi32(p_row1);

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint8_t *ptr_src = img.template ptr<uint8_t>(r);
// 			__m256i v_gain = (r % 2 == 0) ? v_gain_row0 : v_gain_row1;

// 			int c = 0;
// 			// 每次处理 16 个像素 (为了配合 u16 扩展逻辑，保持对齐)
// 			// 虽然 u8 可以 load 32个，但为了计算精度防止溢出，我们按 16个
// 			// 一组处理
// 			for (; c <= cols - 16; c += 16) {
// 				// 1. Load 16x u8 (128-bit)
// 				__m128i v_small =
// 					_mm_loadu_si128((const __m128i *)(ptr_src + c));

// 				// 2. Expand u8 -> u16 (变成 256-bit)
// 				__m256i v_src_16 = _mm256_cvtepu8_epi16(v_small);

// 				// 3. 计算: (Pixel_u16 * Gain_u16) >> 12
// 				// 这里有个技巧：u8(max 255) * Gain(max ~16.0*4096=65536) =
// 				// ~1.6kw 结果可以塞进 int32。

// 				// Unpack Low 8 -> u32
// 				__m256i v_src_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src_16));
// 				__m256i v_gain_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_gain));

// 				// Unpack High 8 -> u32
// 				__m256i v_src_hi = _mm256_cvtepu16_epi32(
// 					_mm256_extracti128_si256(v_src_16, 1));
// 				__m256i v_gain_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_gain, 1));

// 				// Mul & Shift
// 				__m256i v_res_lo = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);
// 				__m256i v_res_hi = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

// 				// 4. Pack u32 -> u16 (Saturate)
// 				__m256i v_packed_16 = _mm256_packus_epi32(v_res_lo, v_res_hi);
// 				// 修正 packus 的乱序
// 				v_packed_16 = _mm256_permute4x64_epi64(v_packed_16,
// 													   _MM_SHUFFLE(3, 1, 2, 0));

// 				// 5. Pack u16 -> u8 (Saturate)
// 				// packus_epi16 把两个 256bit 压成一个
// 				// 256bit，但我们只需要半个(128bit) 所以我们把 v_packed_16
// 				// 拆成两半喂给它
// 				__m128i v_final =
// 					_mm_packus_epi16(_mm256_castsi256_si128(v_packed_16),
// 									 _mm256_extracti128_si256(v_packed_16, 1));

// 				// 6. Store
// 				_mm_storeu_si128((__m128i *)(ptr_src + c), v_final);
// 			}

// 			// Tail
// 			uint16_t gain0 = (r % 2 == 0) ? g_tl : g_bl;
// 			uint16_t gain1 = (r % 2 == 0) ? g_tr : g_br;
// 			for (; c < cols; ++c) {
// 				uint16_t gain = (c % 2 == 0) ? gain0 : gain1;
// 				uint32_t val = ((uint32_t)ptr_src[c] * gain) >> 12;
// 				if (val > 255)
// 					val = 255;
// 				ptr_src[c] = (uint8_t)val;
// 			}
// 		}
// 		return *this;
// 	}

// 	// SIMD 核：u16 Raw -> LSC -> AWB -> u8 Raw
// 	LFIsp &raw_to_8bit_with_gains_simd(cv::Mat &dst_8u, float exposure = 1.5f) {
// 		int rows = lfp_img_.rows;
// 		int cols = lfp_img_.cols;

// 		// ============================================================
// 		// 1. 准备参数 (BLC + Scale + AWB + Exposure)
// 		// ============================================================

// 		uint16_t bl_val = static_cast<uint16_t>(config_.black_level);
// 		__m256i v_bl = _mm256_set1_epi16(bl_val);

// 		// [关键修改 1] 计算综合缩放系数
// 		// 有效范围 = White - Black
// 		float effective_range =
// 			static_cast<float>(config_.white_level - config_.black_level);
// 		if (effective_range < 1.0f)
// 			effective_range = 1.0f;

// 		// 公式解析：
// 		// 1. 归一化: 1.0 / effective_range
// 		// 2. 曝光增益: * exposure
// 		// 3. 映射到 u8: * 255.0f
// 		// 4. 定点化 Q12: * 4096.0f
// 		float total_scale_factor =
// 			(255.0f / effective_range) * exposure * 4096.0f;

// 		// [关键修改 2] 计算最终增益 (包含 Exposure) 并防止 u16 溢出
// 		// 如果 exposure 给很大，导致系数超过
// 		// 65535，必须截断，否则回绕会导致图像花屏
// 		auto calc_gain = [&](float awb_g) -> uint16_t {
// 			float val = awb_g * total_scale_factor;
// 			if (val > 65535.0f)
// 				val = 65535.0f; // 必须做饱和保护！
// 			return static_cast<uint16_t>(val);
// 		};

// 		uint16_t g_tl = calc_gain(config_.awb_gains[0]);
// 		uint16_t g_tr = calc_gain(config_.awb_gains[1]);
// 		uint16_t g_bl = calc_gain(config_.awb_gains[2]);
// 		uint16_t g_br = calc_gain(config_.awb_gains[3]);

// 		// 广播 AWB 向量
// 		uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
// 		__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
// 		uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
// 		__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

// 		bool has_lsc = !lsc_gain_map_int_.empty()
// 					   && lsc_gain_map_int_.size() == lfp_img_.size();

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			const uint16_t *src = lfp_img_.template ptr<uint16_t>(r);
// 			// 如果有 LSC，获取当前行指针；否则 nullptr
// 			const uint16_t *lsc =
// 				has_lsc ? lsc_gain_map_int_.ptr<uint16_t>(r) : nullptr;
// 			uint8_t *dst = dst_8u.ptr<uint8_t>(r);

// 			__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

// 			int c = 0;
// 			for (; c <= cols - 16; c += 16) {
// 				// 1. Load Pixel
// 				__m256i v_src = _mm256_loadu_si256((const __m256i *)(src + c));

// 				// ========================================================
// 				// 【新增步骤】 2. BLC (饱和减法)
// 				// ========================================================
// 				// result = (pixel > bl) ? (pixel - bl) : 0
// 				// 这一步必须在所有增益计算之前！
// 				v_src = _mm256_subs_epu16(v_src, v_bl);

// 				// 3. Load & Apply LSC (if exists)
// 				// LSC (Q12) * Pixel = Result (Q12) -> Shift >> 12
// 				if (has_lsc) {
// 					__m256i v_lsc =
// 						_mm256_loadu_si256((const __m256i *)(lsc + c));

// 					// Unpack low 8
// 					__m256i v_src_lo =
// 						_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
// 					__m256i v_lsc_lo =
// 						_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc));
// 					v_src_lo = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_src_lo, v_lsc_lo), 12);

// 					// Unpack high 8
// 					__m256i v_src_hi = _mm256_cvtepu16_epi32(
// 						_mm256_extracti128_si256(v_src, 1));
// 					__m256i v_lsc_hi = _mm256_cvtepu16_epi32(
// 						_mm256_extracti128_si256(v_lsc, 1));
// 					v_src_hi = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_src_hi, v_lsc_hi), 12);

// 					// Pack back to u16
// 					v_src = _mm256_packus_epi32(v_src_lo, v_src_hi);
// 					v_src = _mm256_permute4x64_epi64(v_src,
// 													 _MM_SHUFFLE(3, 1, 2, 0));
// 				}

// 				// 4. Apply AWB + Scale (Pack to u8)
// 				// Result = (Pixel * AWB_Q12) >> 12

// 				// Low 8
// 				__m256i v_src_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
// 				__m256i v_awb_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
// 				v_src_lo = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_lo, v_awb_lo), 12);

// 				// High 8
// 				__m256i v_src_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));
// 				__m256i v_awb_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));
// 				v_src_hi = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_hi, v_awb_hi), 12);

// 				// Pack u32 -> u16
// 				__m256i v_res_16 = _mm256_packus_epi32(v_src_lo, v_src_hi);
// 				v_res_16 =
// 					_mm256_permute4x64_epi64(v_res_16, _MM_SHUFFLE(3, 1, 2, 0));

// 				// Pack u16 -> u8
// 				__m128i v_res_8 =
// 					_mm_packus_epi16(_mm256_castsi256_si128(v_res_16),
// 									 _mm256_extracti128_si256(v_res_16, 1));

// 				// Store
// 				_mm_storeu_si128((__m128i *)(dst + c), v_res_8);
// 			}

// 			// Tail (标量部分)
// 			uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
// 			uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;
// 			for (; c < cols; ++c) {
// 				uint32_t val = src[c];

// 				// 1. BLC
// 				val = (val > bl_val) ? (val - bl_val) : 0;

// 				// 2. LSC
// 				if (has_lsc)
// 					val = (val * lsc[c]) >> 12;

// 				// 3. AWB + Scale
// 				uint32_t awb = (c % 2 == 0) ? awb_0 : awb_1;
// 				val = (val * awb) >> 12;

// 				if (val > 255)
// 					val = 255;
// 				dst[c] = static_cast<uint8_t>(val);
// 			}
// 		}
// 	}

// 	LFIsp &raw_to_8bit_with_gains_simd_u8(cv::Mat &dst_8u,
// 										  float exposure = 1.5f) {
// 		int rows = lfp_img_.rows;
// 		int cols = lfp_img_.cols;

// 		// ============================================================
// 		// 1. 准备参数 (BLC + Scale + AWB + Exposure)
// 		// ============================================================

// 		// [BLC] 自动适配 8-bit 黑电平
// 		// 如果原始位深是 10-bit (BL=64)，在 8-bit 图里 BL 应该是 16
// 		int bl_shift = (config_.bitDepth > 8) ? (config_.bitDepth - 8) : 0;
// 		uint8_t bl_val = static_cast<uint8_t>(config_.black_level >> bl_shift);
// 		__m256i v_bl = _mm256_set1_epi8(bl_val);

// 		// [Scale] 计算 8-bit 下的有效动态范围
// 		// 8-bit 下，最大值是 255，黑电平是 bl_val
// 		float effective_range = 255.0f - bl_val;
// 		if (effective_range < 1.0f)
// 			effective_range = 1.0f;

// 		// [Total Gain] 综合系数 = (255/Range) * Exposure * 4096 (Q12)
// 		// 我们把所有乘法合并，让 SIMD 内只做一次乘法
// 		float total_scale_factor =
// 			(255.0f / effective_range) * exposure * 4096.0f;

// 		// 辅助 lambda: 计算并防溢出
// 		auto calc_gain = [&](float awb_g) -> uint16_t {
// 			float val = awb_g * total_scale_factor;
// 			if (val > 65535.0f)
// 				val = 65535.0f;
// 			return static_cast<uint16_t>(val);
// 		};

// 		uint16_t g_tl = calc_gain(config_.awb_gains[0]);
// 		uint16_t g_tr = calc_gain(config_.awb_gains[1]);
// 		uint16_t g_bl = calc_gain(config_.awb_gains[2]);
// 		uint16_t g_br = calc_gain(config_.awb_gains[3]);

// 		// 广播 AWB 向量 (u32 包含两个 u16 Gain)
// 		uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
// 		__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);
// 		uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
// 		__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

// 		bool has_lsc = !lsc_gain_map_int_.empty()
// 					   && lsc_gain_map_int_.size() == lfp_img_.size();

// 		// ============================================================
// 		// 2. SIMD Loop (32 pixels per step)
// 		// ============================================================
// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint8_t *src = lfp_img_.ptr<uint8_t>(r);
// 			uint8_t *dst = dst_8u.ptr<uint8_t>(r);
// 			const uint16_t *lsc =
// 				has_lsc ? lsc_gain_map_int_.ptr<uint16_t>(r) : nullptr;

// 			__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

// 			int c = 0;
// 			for (; c <= cols - 32; c += 32) {
// 				// 1. Load 32 Pixels (u8) -> 1个 YMM
// 				__m256i v_src_32 =
// 					_mm256_loadu_si256((const __m256i *)(src + c));

// 				// 2. BLC (8-bit 饱和减法)
// 				v_src_32 = _mm256_subs_epu8(v_src_32, v_bl);

// 				// 3. 准备 LSC (2个 YMM)
// 				__m256i v_lsc_0, v_lsc_1;
// 				if (has_lsc) {
// 					v_lsc_0 = _mm256_loadu_si256((const __m256i *)(lsc + c));
// 					v_lsc_1 =
// 						_mm256_loadu_si256((const __m256i *)(lsc + c + 16));
// 				} else {
// 					// 如果没有 LSC，默认为 4096 (1.0 in Q12)
// 					__m256i v_one = _mm256_set1_epi16(4096);
// 					v_lsc_0 = v_one;
// 					v_lsc_1 = v_one;
// 				}

// 				// -----------------------------------------------------------
// 				// 核心计算 Lambda: 处理 16 个像素
// 				// Input: v_p_8 (128bit u8 part), v_lsc_16 (256bit u16)
// 				// Output: 256bit u16 result
// 				// -----------------------------------------------------------
// 				auto process_half = [&](__m128i v_p_8,
// 										__m256i v_lsc_16) -> __m256i {
// 					// Unpack Pixel u8 -> u32 (Low 8)
// 					__m256i v_p_lo = _mm256_cvtepu8_epi32(v_p_8);
// 					// Unpack Pixel u8 -> u32 (High 8)
// 					__m256i v_p_hi =
// 						_mm256_cvtepu8_epi32(_mm_unpackhi_epi64(v_p_8, v_p_8));

// 					// Unpack LSC u16 -> u32
// 					__m256i v_lsc_lo =
// 						_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc_16));
// 					__m256i v_lsc_hi = _mm256_cvtepu16_epi32(
// 						_mm256_extracti128_si256(v_lsc_16, 1));

// 					// Unpack AWB u16 -> u32 (复用 v_awb)
// 					__m256i v_awb_lo =
// 						_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
// 					__m256i v_awb_hi = _mm256_cvtepu16_epi32(
// 						_mm256_extracti128_si256(v_awb, 1));

// 					// --- 1. 计算 Total Gain = (LSC * AWB) >> 12 ---
// 					// 注意：AWB 已经包含了 Exposure 和
// 					// Scale，所以这里算出来就是最终系数
// 					__m256i v_gain_lo = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), 12);
// 					__m256i v_gain_hi = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), 12);

// 					// --- 2. 应用 Gain = (Pixel * TotalGain) >> 12 ---
// 					v_p_lo = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_p_lo, v_gain_lo), 12);
// 					v_p_hi = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_p_hi, v_gain_hi), 12);

// 					// --- 3. Pack to u16 ---
// 					__m256i v_res_16 = _mm256_packus_epi32(v_p_lo, v_p_hi);
// 					return _mm256_permute4x64_epi64(v_res_16,
// 													_MM_SHUFFLE(3, 1, 2, 0));
// 				};

// 				// 4. 处理前 16 个像素
// 				__m256i v_res_0 =
// 					process_half(_mm256_castsi256_si128(v_src_32), v_lsc_0);

// 				// 5. 处理后 16 个像素
// 				__m256i v_res_1 = process_half(
// 					_mm256_extracti128_si256(v_src_32, 1), v_lsc_1);

// 				// 6. 合并 32 个结果 (u16 -> u8)
// 				__m256i v_res_u8 = _mm256_packus_epi16(v_res_0, v_res_1);
// 				v_res_u8 =
// 					_mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));

// 				// 7. 存储
// 				_mm256_storeu_si256((__m256i *)(dst + c), v_res_u8);
// 			}

// 			// Tail Process (Scalar)
// 			uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
// 			uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;

// 			for (; c < cols; ++c) {
// 				uint32_t val = src[c];

// 				// 1. BLC
// 				val = (val > bl_val) ? (val - bl_val) : 0;

// 				// 2. LSC
// 				if (has_lsc)
// 					val = (val * lsc[c]) >> 12;

// 				// 3. AWB + Scale + Exp
// 				uint32_t awb = (c % 2 == 0) ? awb_0 : awb_1;
// 				val = (val * awb) >> 12;

// 				if (val > 255)
// 					val = 255;
// 				dst[c] = static_cast<uint8_t>(val);
// 			}
// 		}
// 	}

// 	LFIsp &prepare_ccm_fixed_point() {
// 		if (config_.ccm_matrix.empty())
// 			return *this;

// 		ccm_matrix_int_.resize(9);
// 		const float scale = 4096.0f; // Q12 格式

// 		for (int i = 0; i < 9; ++i) {
// 			// 放大并取整
// 			ccm_matrix_int_[i] =
// 				static_cast<int32_t>(config_.ccm_matrix[i] * scale);
// 		}
// 		return *this;
// 	}

// 	LFIsp &lsc_awb_simd_u16(cv::Mat &img) {
// 		int rows = lfp_img_.rows;
// 		int cols = img.cols;

// 		// --- 1. 准备 AWB 增益向量 (Q12 定点数) ---
// 		// 我们把 AWB 增益也放大 4096 倍，跟 LSC 保持一致
// 		const float scale = 4096.0f;
// 		uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
// 		uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
// 		uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
// 		uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

// 		// 构造广播向量 (Row 0: TL/TR, Row 1: BL/BR)
// 		uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
// 		__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);

// 		uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
// 		__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint16_t *ptr_src = img.template ptr<uint16_t>(r);
// 			const uint16_t *ptr_lsc = lsc_gain_map_int_.ptr<uint16_t>(r);

// 			// 选择当前行的 AWB 向量
// 			__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

// 			int c = 0;
// 			// 每次处理 16 个像素
// 			for (; c <= cols - 16; c += 16) {
// 				// 1. 加载 LSC Gain (u16, Q12)
// 				__m256i v_lsc =
// 					_mm256_loadu_si256((const __m256i *)(ptr_lsc + c));

// 				// 2. 计算综合增益 Total_Gain = (LSC * AWB) >> 12
// 				// LSC(Q12) * AWB(Q12) = Q24 -> 右移 12 -> Q12
// 				// 这样可以保证精度，同时结果仍在 u16 范围内
// 				// (只要总增益不超过 16.0)

// 				// Unpack LSC to 32-bit
// 				__m256i v_lsc_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc));
// 				__m256i v_lsc_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_lsc, 1));

// 				// Unpack AWB to 32-bit
// 				__m256i v_awb_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
// 				__m256i v_awb_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_awb, 1));

// 				// Multiply & Shift (计算综合增益)
// 				__m256i v_gain_lo = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), 12);
// 				__m256i v_gain_hi = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), 12);

// 				// 3. 加载像素
// 				__m256i v_src =
// 					_mm256_loadu_si256((const __m256i *)(ptr_src + c));
// 				__m256i v_src_lo =
// 					_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_src));
// 				__m256i v_src_hi =
// 					_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v_src, 1));

// 				// 4. 应用综合增益 Result = (Pixel * Total_Gain) >> 12
// 				v_src_lo = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_lo, v_gain_lo), 12);
// 				v_src_hi = _mm256_srli_epi32(
// 					_mm256_mullo_epi32(v_src_hi, v_gain_hi), 12);

// 				// 5. Pack & Saturate (带饱和截断)
// 				__m256i v_res = _mm256_packus_epi32(v_src_lo, v_src_hi);
// 				v_res =
// 					_mm256_permute4x64_epi64(v_res, _MM_SHUFFLE(3, 1, 2, 0));

// 				// 6. Store
// 				_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res);
// 			}

// 			// 处理尾部 (Scalar Fallback)
// 			uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl; // 偶数列增益
// 			uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br; // 奇数列增益

// 			for (; c < cols; ++c) {
// 				uint16_t lsc = ptr_lsc[c];
// 				uint16_t awb = (c % 2 == 0) ? awb_0 : awb_1;

// 				// 计算综合增益 (u32 防止溢出)
// 				uint32_t total_gain = ((uint32_t)lsc * awb) >> 12;

// 				// 应用增益
// 				uint32_t val = ((uint32_t)ptr_src[c] * total_gain) >> 12;

// 				if (val > 65535)
// 					val = 65535;
// 				ptr_src[c] = static_cast<uint16_t>(val);
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &lsc_awb_simd_u8(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// 		const float scale = 4096.0f; // Q12
// 		// 1. 准备 AWB 增益
// 		uint16_t g_tl = static_cast<uint16_t>(config_.awb_gains[0] * scale);
// 		uint16_t g_tr = static_cast<uint16_t>(config_.awb_gains[1] * scale);
// 		uint16_t g_bl = static_cast<uint16_t>(config_.awb_gains[2] * scale);
// 		uint16_t g_br = static_cast<uint16_t>(config_.awb_gains[3] * scale);

// 		// 构造广播向量 (Row 0: TL/TR, Row 1: BL/BR)
// 		// 每个 u32 包含两个增益: [TR | TL]
// 		uint32_t p_row0 = (static_cast<uint32_t>(g_tr) << 16) | g_tl;
// 		__m256i v_awb_row0 = _mm256_set1_epi32(p_row0);

// 		uint32_t p_row1 = (static_cast<uint32_t>(g_br) << 16) | g_bl;
// 		__m256i v_awb_row1 = _mm256_set1_epi32(p_row1);

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint8_t *ptr_src = img.template ptr<uint8_t>(r);
// 			const uint16_t *ptr_lsc = lsc_gain_map_int_.ptr<uint16_t>(r);

// 			// 选择当前行的 AWB 向量 (覆盖 16 像素， pattern 循环)
// 			__m256i v_awb = (r % 2 == 0) ? v_awb_row0 : v_awb_row1;

// 			int c = 0;
// 			// 【优化】：步长改为 32
// 			for (; c <= cols - 32; c += 32) {
// 				// 1. 加载 32 个像素 (u8) -> 1个 YMM 寄存器
// 				__m256i v_src_32 =
// 					_mm256_loadu_si256((const __m256i *)(ptr_src + c));

// 				// 2. 加载 32 个 LSC 增益 (u16) -> 2个 YMM 寄存器
// 				__m256i v_lsc_0 =
// 					_mm256_loadu_si256((const __m256i *)(ptr_lsc + c)); // 0-15
// 				__m256i v_lsc_1 = _mm256_loadu_si256(
// 					(const __m256i *)(ptr_lsc + c + 16)); // 16-31

// 				// -------------------------------------------------------------
// 				// 定义计算核心 Lambda (处理 16 个像素)
// 				// Input: v_p_8 (128bit u8 part), v_lsc_16 (256bit u16), v_awb
// 				// (256bit) Output: 256bit u16 result
// 				// -------------------------------------------------------------
// 				auto process_half = [&](__m128i v_p_8,
// 										__m256i v_lsc_16) -> __m256i {
// 					// 像素 u8 -> u32 (Low 8)
// 					__m256i v_p_lo = _mm256_cvtepu8_epi32(v_p_8);
// 					// 像素 u8 -> u32 (High 8)
// 					__m256i v_p_hi =
// 						_mm256_cvtepu8_epi32(_mm_unpackhi_epi64(v_p_8, v_p_8));

// 					// LSC u16 -> u32 (Low 8)
// 					__m256i v_lsc_lo =
// 						_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_lsc_16));
// 					// LSC u16 -> u32 (High 8)
// 					__m256i v_lsc_hi = _mm256_cvtepu16_epi32(
// 						_mm256_extracti128_si256(v_lsc_16, 1));

// 					// AWB u16 -> u32 (Low 8)
// 					__m256i v_awb_lo =
// 						_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v_awb));
// 					// AWB u16 -> u32 (High 8)
// 					__m256i v_awb_hi = _mm256_cvtepu16_epi32(
// 						_mm256_extracti128_si256(v_awb, 1));

// 					// --- 计算 Total Gain = (LSC * AWB) >> 12 ---
// 					__m256i v_gain_lo = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_lsc_lo, v_awb_lo), 12);
// 					__m256i v_gain_hi = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_lsc_hi, v_awb_hi), 12);

// 					// --- 应用 Gain = (Pixel * TotalGain) >> 12 ---
// 					v_p_lo = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_p_lo, v_gain_lo), 12);
// 					v_p_hi = _mm256_srli_epi32(
// 						_mm256_mullo_epi32(v_p_hi, v_gain_hi), 12);

// 					// Pack u32 -> u16 (此时是 16 个 u16)
// 					__m256i v_res_16 = _mm256_packus_epi32(v_p_lo, v_p_hi);
// 					// 修正 lane 顺序
// 					return _mm256_permute4x64_epi64(v_res_16,
// 													_MM_SHUFFLE(3, 1, 2, 0));
// 				};

// 				// 3. 处理前 16 个像素
// 				// v_src_32 的低 128 位
// 				__m256i v_res_0 =
// 					process_half(_mm256_castsi256_si128(v_src_32), v_lsc_0);

// 				// 4. 处理后 16 个像素
// 				// v_src_32 的高 128 位
// 				__m256i v_res_1 = process_half(
// 					_mm256_extracti128_si256(v_src_32, 1), v_lsc_1);

// 				// 5. Pack u16 -> u8 (合并 32 个结果)
// 				// _mm256_packus_epi16 将两个 256bit (u16) 压缩为一个 256bit
// 				// (u8)
// 				__m256i v_res_u8 = _mm256_packus_epi16(v_res_0, v_res_1);

// 				// 6. 修正最终的 Lane 顺序
// 				// AVX2 的 pack 是 lane-wise 的: [A_lo, B_lo, A_hi, B_hi]
// 				// 我们需要: [A_lo, A_hi, B_lo, B_hi] (其中 A 是前16px, B
// 				// 是后16px)
// 				v_res_u8 =
// 					_mm256_permute4x64_epi64(v_res_u8, _MM_SHUFFLE(3, 1, 2, 0));

// 				// 7. 存储 32 字节
// 				_mm256_storeu_si256((__m256i *)(ptr_src + c), v_res_u8);
// 			}

// 			// 标量部分 (处理剩余像素)
// 			uint16_t awb_0 = (r % 2 == 0) ? g_tl : g_bl;
// 			uint16_t awb_1 = (r % 2 == 0) ? g_tr : g_br;

// 			for (; c < cols; ++c) {
// 				uint16_t awb = (c % 2 == 0) ? awb_0 : awb_1;
// 				uint16_t lsc = ptr_lsc[c];
// 				uint32_t total_gain = ((uint32_t)lsc * awb) >> 12;
// 				uint32_t val = ((uint32_t)ptr_src[c] * total_gain) >> 12;
// 				if (val > 255)
// 					val = 255;
// 				ptr_src[c] = static_cast<uint8_t>(val);
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &ccm_fixed_u16(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// 		// 提取矩阵系数到局部变量 (寄存器缓存)
// 		const int32_t c00 = ccm_matrix_int_[0], c01 = ccm_matrix_int_[1],
// 					  c02 = ccm_matrix_int_[2];
// 		const int32_t c10 = ccm_matrix_int_[3], c11 = ccm_matrix_int_[4],
// 					  c12 = ccm_matrix_int_[5];
// 		const int32_t c20 = ccm_matrix_int_[6], c21 = ccm_matrix_int_[7],
// 					  c22 = ccm_matrix_int_[8];

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint16_t *ptr = img.template ptr<uint16_t>(r);

// 			// 每次处理 1 个像素 (3个通道)
// 			// 编译器会自动对这种简单的整数计算进行 Loop Unrolling 和 SIMD 优化
// 			for (int c = 0; c < cols; ++c) {
// 				int idx = c * 3;

// 				// 读取原始 RGB (提升为 int32 防止乘法溢出)
// 				int32_t r_val =
// 					ptr[idx + 0]; // 假设 BGR or RGB
// 								  // 顺序不影响数学计算，只影响系数对应
// 				int32_t g_val = ptr[idx + 1];
// 				int32_t b_val = ptr[idx + 2];

// 				// 核心计算: 矩阵乘法 + 右移 12 位
// 				// 使用局部变量累加，减少内存读写
// 				int32_t new_ch0 =
// 					(r_val * c00 + g_val * c01 + b_val * c02) >> 12;
// 				int32_t new_ch1 =
// 					(r_val * c10 + g_val * c11 + b_val * c12) >> 12;
// 				int32_t new_ch2 =
// 					(r_val * c20 + g_val * c21 + b_val * c22) >> 12;

// 				// 饱和截断 (Saturate)
// 				// 手写比较比 std::clamp 更快 (无异常检查)
// 				if (new_ch0 < 0)
// 					new_ch0 = 0;
// 				else if (new_ch0 > 65535)
// 					new_ch0 = 65535;
// 				if (new_ch1 < 0)
// 					new_ch1 = 0;
// 				else if (new_ch1 > 65535)
// 					new_ch1 = 65535;
// 				if (new_ch2 < 0)
// 					new_ch2 = 0;
// 				else if (new_ch2 > 65535)
// 					new_ch2 = 65535;

// 				// 写入
// 				ptr[idx + 0] = static_cast<uint16_t>(new_ch0);
// 				ptr[idx + 1] = static_cast<uint16_t>(new_ch1);
// 				ptr[idx + 2] = static_cast<uint16_t>(new_ch2);
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &ccm_fixed_u8(cv::Mat &img) {
// 		int rows = img.rows;
// 		int cols = img.cols;

// 		const int32_t c00 = ccm_matrix_int_[0], c01 = ccm_matrix_int_[1],
// 					  c02 = ccm_matrix_int_[2];
// 		const int32_t c10 = ccm_matrix_int_[3], c11 = ccm_matrix_int_[4],
// 					  c12 = ccm_matrix_int_[5];
// 		const int32_t c20 = ccm_matrix_int_[6], c21 = ccm_matrix_int_[7],
// 					  c22 = ccm_matrix_int_[8];

// #pragma omp parallel for
// 		for (int r = 0; r < rows; ++r) {
// 			uint8_t *ptr = img.template ptr<uint8_t>(r);

// 			for (int c = 0; c < cols; ++c) {
// 				int idx = c * 3;
// 				int32_t r_val = ptr[idx];
// 				int32_t g_val = ptr[idx + 1];
// 				int32_t b_val = ptr[idx + 2];

// 				int32_t new_ch0 =
// 					(r_val * c00 + g_val * c01 + b_val * c02) >> 12;
// 				int32_t new_ch1 =
// 					(r_val * c10 + g_val * c11 + b_val * c12) >> 12;
// 				int32_t new_ch2 =
// 					(r_val * c20 + g_val * c21 + b_val * c22) >> 12;

// 				if (new_ch0 < 0)
// 					new_ch0 = 0;
// 				else if (new_ch0 > 255)
// 					new_ch0 = 255;
// 				if (new_ch1 < 0)
// 					new_ch1 = 0;
// 				else if (new_ch1 > 255)
// 					new_ch1 = 255;
// 				if (new_ch2 < 0)
// 					new_ch2 = 0;
// 				else if (new_ch2 > 255)
// 					new_ch2 = 255;

// 				ptr[idx] = (uint8_t)new_ch0;
// 				ptr[idx + 1] = (uint8_t)new_ch1;
// 				ptr[idx + 2] = (uint8_t)new_ch2;
// 			}
// 		}
// 		return *this;
// 	}

// 	LFIsp &compute_lab_stats(const cv::Mat &src, cv::Scalar &mean,
// 							 cv::Scalar &stddev) {
// 		cv::Mat lab;

// 		// --- 修复开始 ---
// 		int depth = src.depth();

// 		// 情况 1: 8-bit 图像 (0-255) -> 归一化除以 255
// 		if (depth == CV_8U) {
// 			src.convertTo(lab, CV_32F, 1.0 / 255.0);
// 		}
// 		// 情况 2: 16-bit 图像 (CV_16U) -> 必须转为 CV_32F 才能进行 Lab 转换
// 		else if (depth == CV_16U) {
// 			// 如果你的类能访问到 bitDepth 参数 (比如 m_bitDepth)，建议用它：
// 			// double scale = 1.0 / ((1 << m_bitDepth) - 1);

// 			// 如果访问不到，最安全的通用做法是按 16位最大值归一化：
// 			src.convertTo(lab, CV_32F, 1.0 / 65535.0);
// 		}
// 		// 情况 3: 已经是浮点型 (CV_32F / CV_64F)
// 		else if (depth == CV_32F || depth == CV_64F) {
// 			lab = src;
// 		} else {
// 			// 其他不支持的类型，转为 32F 并假设是 8位 (兜底)
// 			src.convertTo(lab, CV_32F);
// 		}
// 		// --- 修复结束 ---

// 		// 现在 lab 肯定是 CV_32F 且范围在 [0, 1] 之间，可以安全转换了
// 		// 转到 Lab 空间 (L: 0-100, a/b: -127..127)
// 		cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);

// 		// OpenCV 自带的高效统计函数
// 		cv::meanStdDev(lab, mean, stddev);

// 		return *this;
// 	}

// 	/**
// 	 * @brief 核心算法：应用 Reinhard 变换
// 	 * * @param target  待修改的图像 (Source)
// 	 * @param ref_mean 参考图均值
// 	 * @param ref_std  参考图标准差
// 	 */
// 	LFIsp &apply_reinhard_transfer(cv::Mat &target, const cv::Scalar &ref_mean,
// 								   const cv::Scalar &ref_std) {
// 		if (target.empty())
// 			return *this;

// 		// 1. 转换到 Lab 空间 (Float)
// 		cv::Mat lab;
// 		if (target.depth() == CV_8U) {
// 			target.convertTo(lab, CV_32F, 1.0 / 255.0);
// 		} else {
// 			target.convertTo(lab, CV_32F);
// 		}
// 		cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);

// 		// 2. 计算当前图像的统计信息
// 		cv::Scalar src_mean, src_std;
// 		cv::meanStdDev(lab, src_mean, src_std);

// 		// 3. 分通道应用线性变换
// 		// 公式: Dst = (Src - Src_Mean) * (Ref_Std / Src_Std) + Ref_Mean
// 		// 展开: Dst = Src * alpha + beta
// 		// 其中 alpha = Ref_Std / Src_Std
// 		//      beta  = Ref_Mean - Src_Mean * alpha

// 		std::vector<cv::Mat> channels;
// 		cv::split(lab, channels);

// 		for (int i = 0; i < 3; ++i) {
// 			// 防止除以 0 (处理纯色图片时的边缘情况)
// 			double s_std = (src_std[i] < 1e-6) ? 1e-6 : src_std[i];

// 			double alpha = ref_std[i] / s_std;
// 			double beta = ref_mean[i] - alpha * src_mean[i];

// 			// 使用 OpenCV 优化的线性变换算子
// 			channels[i].convertTo(channels[i], -1, alpha, beta);
// 		}

// 		// 4. 合并通道并转回 RGB
// 		cv::merge(channels, lab);
// 		cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);

// 		// 5. 转回原始数据类型 (通常是 uint8)
// 		if (target.depth() == CV_8U) {
// 			lab.convertTo(target, CV_8U, 255.0);
// 		} else {
// 			lab.copyTo(target);
// 		}

// 		return *this;
// 	}
// };

// #endif // ISP_H
