#ifndef LFISP_H
#define LFISP_H

#include "colormatcher.h"
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

enum class DpcMethod { Diretional };
enum class DemosaicMethod { Bilinear, Gray, VGN, EA };

struct IspConfig {
	BayerPattern bayer = BayerPattern::NONE;
	int bitDepth = 8;
	int white_level = 255, black_level = 0;
	std::vector<float> awb_gains = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> ccm_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
									 0.0f, 0.0f, 0.0f, 1.0f};
	float gamma = 1.0f;

	int dpcThreshold = 25;
	float lscExp = 1.0;
	bool enableDPC = true;
	bool enableBLC = true;
	bool enableLSC = true;
	bool enableAWB = true;
	bool enableDemosaic = true;
	bool enableCCM = true;
	bool enableGamma = true;
	bool enableExtract = true;
	bool enableDehex = true;
	bool enableColorEq = true;
	DpcMethod dpcMethod = DpcMethod::Diretional;
	DemosaicMethod demosaicMethod = DemosaicMethod::Bilinear;
	ColorEqualizeMethod colorEqMethod = ColorEqualizeMethod::Reinhard;
};

class LFIsp {
public:
	struct ResampleMaps {
		std::vector<cv::Mat> extract;
		std::vector<cv::Mat> dehex;
	} maps;

	explicit LFIsp();
	explicit LFIsp(const cv::Mat &lfp_img);
	explicit LFIsp(const cv::Mat &lfp_img, const cv::Mat &wht_img,
				   const IspConfig &config);

	cv::Mat &getResult() { return lfp_img_; }
	const cv::Mat &getResult() const { return lfp_img_; }
	const std::vector<cv::Mat> &getSAIS() const { return sais; }
	std::vector<cv::Mat> &getSAIS() { return sais; }
	bool isLutEmpty() { return maps.extract.empty() || maps.dehex.empty(); }

	LFIsp &print_config(const IspConfig &config);

	static std::string bayerToString(BayerPattern p);
	static void parseJsonToConfig(const json &j, IspConfig &config);

	LFIsp &set_lf_img(const cv::Mat &img);
	LFIsp &initConfig(const cv::Mat &img, const IspConfig &config);

	LFIsp &blc(int black_level);
	LFIsp &dpc(int threshold = 100);
	LFIsp &lsc(float exposure);
	LFIsp &awb(const std::vector<float> &wbgains);
	LFIsp &ccm(const std::vector<float> &ccm_matrix);
	LFIsp &demosaic(BayerPattern bayer, DemosaicMethod method); // TODO

	LFIsp &blc_fast(int black_level);
	LFIsp &dpc_fast(DpcMethod method, int threshold = 100); // TODO
	LFIsp &lsc_fast(float exposure);
	LFIsp &awb_fast(const std::vector<float> &wbgains);
	LFIsp &lsc_awb_fused_fast(float exposure,
							  const std::vector<float> &wbgains);
	LFIsp &resample(bool dehex);
	LFIsp &ccm_fast(const std::vector<float> &ccm_matrix);
	LFIsp &gc_fast(float gamma, int bitDepth);

	LFIsp &preview(const IspConfig &config);
	LFIsp &process(const IspConfig &config);

private:
	cv::Mat lfp_img_;
	std::vector<cv::Mat> sais;

	cv::Mat lsc_gain_map_, lsc_gain_map_int_;
	std::vector<float> last_ccm_matrix_;
	std::vector<int32_t> ccm_matrix_int_;
	std::vector<uint16_t> gamma_lut_u16;
	cv::Mat gamma_lut_u8;
	float last_gamma_ = -1.0f; // 初始值设为负数，确保第一次必定触发生成
	int last_bit_depth_ = -1;  // 初始值设为 -1

	static constexpr int FIXED_BITS = 10;		  // 位移量
	static constexpr float FIXED_SCALE = 1024.0f; // 缩放因子 (1 << 10)

private:
	void prepare_lsc_maps(const cv::Mat &raw_wht, int black_level);
	void prepare_gamma_lut(float gamma, int bitDepth);
	void prepare_ccm_fixed_point(const std::vector<float> &matrix);

	// SIMD 实现的具体版本
	LFIsp &blc_simd_u16(cv::Mat &img, int black_level);
	LFIsp &blc_simd_u8(cv::Mat &img, int black_level);

	LFIsp &dpc_simd_u16(cv::Mat &img, DpcMethod method, int threshold);
	LFIsp &dpc_simd_u8(cv::Mat &img, DpcMethod method, int threshold);

	LFIsp &lsc_simd_u16(cv::Mat &img, float exposure);
	LFIsp &lsc_simd_u8(cv::Mat &img, float exposure);

	LFIsp &awb_simd_u16(cv::Mat &img, const std::vector<float> &wbgains);
	LFIsp &awb_simd_u8(cv::Mat &img, const std::vector<float> &wbgains);

	LFIsp &lsc_awb_simd_u16(cv::Mat &img, float exposure,
							const std::vector<float> &wbgains);
	LFIsp &lsc_awb_simd_u8(cv::Mat &img, float exposure,
						   const std::vector<float> &wbgains);

	LFIsp &raw_to_8bit_with_gains_simd_u16(cv::Mat &dst_8u,
										   const IspConfig &config);
	LFIsp &raw_to_8bit_with_gains_simd_u8(cv::Mat &dst_8u,
										  const IspConfig &config);

	LFIsp &ccm_fixed_u16(cv::Mat &img);
	LFIsp &ccm_fixed_u8(cv::Mat &img);
};

#endif // LFISP_H
