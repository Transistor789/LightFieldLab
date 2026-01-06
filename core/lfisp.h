#ifndef LFISP_H
#define LFISP_H

#include "json.hpp"
#include "utils.h"

#include <immintrin.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using json = nlohmann::json;

enum class BayerPattern { NONE, RGGB, GRBG, GBRG, BGGR };

class LFIsp {
public:
	struct ResampleMaps {
		std::vector<cv::Mat> extract;
		std::vector<cv::Mat> dehex;
	} maps;

public:
	struct Method {
		enum class DPC { Diretional };
		enum class Demosaic { Bilinear, Gray, VGN, EA };
	};
	struct IspConfig {
		BayerPattern bayer = BayerPattern::NONE;
		int width;
		int height;
		int bitDepth = 8;
		int dpcThreshold = 25;
		int white_level = 255, black_level = 0;
		float lscExp = 1.0;
		std::vector<float> awb_gains = {1.0f, 1.0f, 1.0f, 1.0f};
		std::vector<float> ccm_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
										 0.0f, 0.0f, 0.0f, 1.0f};
		float gamma = 1.0f;
	};

	explicit LFIsp();
	explicit LFIsp(const IspConfig &config, const cv::Mat &lfp_img,
				   const cv::Mat &wht_img);
	explicit LFIsp(const json &json_config, const cv::Mat &lfp_img,
				   const cv::Mat &wht_img);

	IspConfig getConfig() const { return config_; }
	cv::Mat &getResult() { return lfp_img_; }
	const cv::Mat &getPreviewResult() const { return preview_img_; }
	const cv::Mat &getResult() const { return lfp_img_; }
	const std::vector<cv::Mat> &getSAIS() const { return sais; }
	std::vector<cv::Mat> &getSAIS() { return sais; }

	IspConfig &get_config() { return config_; }

	LFIsp &set_config(const IspConfig &new_config);
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
	IspConfig config_;
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
	int get_demosaic_code(BayerPattern pattern, bool gray = false);

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
