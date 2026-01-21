#ifndef LFISP_H
#define LFISP_H

#include "colormatcher.h"
#include "hexgrid_fit.h"
#include "json.hpp"
#include "utils.h"

#include <immintrin.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using json = nlohmann::json;

enum class DpcMethod { Diretional };
enum class Device { CPU, CPU_OPENMP_SIMD, GPU };
static constexpr int FIXED_BITS = 10;		  // 位移量
static constexpr float FIXED_SCALE = 1024.0f; // 缩放因子 (1 << 10)

struct IspConfig {
	BayerPattern bayer = BayerPattern::NONE;
	int bitDepth = 8;
	int white_level = 255, black_level = 0;
	std::vector<float> awb_gains = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> ccm_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	float gamma = 1.0f;

	DpcMethod dpcMethod = DpcMethod::Diretional;
	int dpcThreshold = 25;
	float rawnr_sigma_s = 1.5f;	 // 空间域：对应约 5x5 窗口
	float rawnr_sigma_r = 15.0f; // 值域：保护边缘，滤除中低强度噪点
	float lscExp = 1.0;
	float uvnr_sigma_s = 3.0f;	// 空间域：色度斑块通常是低频的，半径需略大
	float uvnr_sigma_r = 20.0f; // 值域：色度边缘较缓，可放宽过滤阈值以抹除色斑
	ColorEqualizeMethod colorEqMethod = ColorEqualizeMethod::Reinhard;
	float ceClipLimit = 2.0f;
	int ceGridSize = 8;
	float seFactor = 1.2f;

	bool enableRAW = true;
	bool enableBLC = true;
	bool enableDPC = true;
	bool enableRawNR = true;
	bool enableLSC = true;
	bool enableAWB = true;
	bool enableDemosaic = true;
	bool enableCCM = true;
	bool enableGamma = true;
	bool enableCSC = true;
	bool enableUVNR = true;
	bool enableColorEq = true;
	bool enableCE = true;
	bool enableSE = true;
	bool enableExtract = true;
	bool enableDehex = true;
	bool benchmark = false;

	Device device = Device::CPU;
};

class LFIsp {
public:
	struct ResampleMaps {
		std::vector<cv::Mat> extract;
		std::vector<cv::Mat> dehex;
	} maps;

	explicit LFIsp();
	explicit LFIsp(const cv::Mat &lfp_img);
	explicit LFIsp(const cv::Mat &lfp_img, const cv::Mat &wht_img, const IspConfig &config);

	cv::Mat &getResult() { return lfp_img_; }
	const cv::Mat &getResult() const { return lfp_img_; }
	const std::vector<cv::Mat> &getSAIs() const { return sais; }
	std::vector<cv::Mat> &getSAIs() { return sais; }
	bool isLutEmpty() { return maps.extract.empty() || maps.dehex.empty(); }

	LFIsp &print_config(const IspConfig &config);

	static std::string bayerToString(BayerPattern p);
	static void parseJsonToConfig(const json &j, IspConfig &config);

	LFIsp &set_lf_img(const cv::Mat &img);
	LFIsp &initConfig(const cv::Mat &img, const IspConfig &config);

	// baseline
	LFIsp &blc(int black_level, int white_level);
	LFIsp &dpc(int threshold = 100);
	LFIsp &lsc(float exposure);
	LFIsp &rawnr(float sigma_spatial, float sigma_color);
	LFIsp &awb(const std::vector<float> &wbgains);
	LFIsp &demosaic(BayerPattern bayer);
	LFIsp &resample(bool dehex);
	LFIsp &ccm(const std::vector<float> &ccm_matrix);
	LFIsp &gc(float gamma);
	LFIsp &csc();
	LFIsp &uvnr(float h_sigma_s, float h_sigma_r);
	LFIsp &color_eq(ColorEqualizeMethod method);
	LFIsp &ce(float clipLimit = 2.0f, int gridSize = 8);
	LFIsp &se(float factor = 1.2f);
	LFIsp &process(const IspConfig &config);

	// openmp+simd accelerated
	LFIsp &blc_fast(int black_level, int white_level);
	LFIsp &dpc_fast(DpcMethod method, int threshold = 100);
	LFIsp &rawnr_fast(float sigma_spatial, float sigma_color);
	LFIsp &lsc_fast(float exposure);
	LFIsp &awb_fast(const std::vector<float> &wbgains);
	LFIsp &lsc_awb_fused_fast(float exposure, const std::vector<float> &wbgains);
	LFIsp &ccm_fast(const std::vector<float> &ccm_matrix);
	LFIsp &gc_fast(float gamma, int bitDepth);
	LFIsp &uvnr_fast(float h_sigma_s, float h_sigma_r);
	LFIsp &ce_fast(float clipLimit = 2.0f, int gridSize = 8);
	LFIsp &se_fast(float factor = 1.2f);

	LFIsp &preview(const IspConfig &config);
	LFIsp &process_fast(const IspConfig &config);

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

private:
	void prepare_lsc_maps(const cv::Mat &raw_wht, int black_level);
	void prepare_ccm_fixed_point(const std::vector<float> &matrix);
	void prepare_gamma_lut(float gamma, int bitDepth);

public: // gpu
	LFIsp &set_lf_gpu(const cv::Mat &img);
	cv::Mat getResultGpu();
	std::vector<cv::Mat> getSAIsGpu();
	void update_resample_maps();

	LFIsp &blc_gpu(int black_level, int white_level);
	LFIsp &dpc_gpu(int threshold);
	LFIsp &nr_gpu(float sigma_spatial, float sigma_color);
	LFIsp &lsc_gpu(float exposure);
	LFIsp &awb_gpu(const std::vector<float> &wbgains);
	LFIsp &lsc_awb_fused_gpu(float exposure, const std::vector<float> &wbgains);
	LFIsp &demosaic_gpu(BayerPattern bayer);
	LFIsp &ccm_gpu(const std::vector<float> &ccm_matrix);
	LFIsp &gc_gpu(float gamma);
	LFIsp &csc_gpu();
	LFIsp &color_eq_gpu(ColorEqualizeMethod method);
	LFIsp &uvnr_gpu(float h_sigma_s, float h_sigma_r);
	LFIsp &ce_gpu(float clipLimit = 2.0f, int gridSize = 8);
	LFIsp &se_gpu(float factor = 1.2f);
	LFIsp &ccm_gamma_fused_gpu(const std::vector<float> &ccm_matrix, float gamma);
	LFIsp &resample_gpu(bool dehex);
	LFIsp &process_gpu(const IspConfig &config);
	LFIsp &global_resample_gpu(std::shared_ptr<HexGridFitter> fitter, bool hex_stretch = true);

private:
	cv::cuda::GpuMat lfp_img_gpu, lsc_map_gpu;
	cv::cuda::Stream stream;
	std::vector<cv::cuda::GpuMat> sais_gpu;
	std::vector<cv::cuda::GpuMat> extract_maps_gpu;
	std::vector<cv::cuda::GpuMat> dehex_maps_gpu;
	cv::Mat _global_tra_mat;

public:
	struct Profiler {
		std::vector<std::pair<std::string, double>> stats;
		int run_count = 0;

		void reset() {
			stats.clear();
			run_count = 0;
		}

		void add(const std::string &name, double ms) {
			// 修改 2: 手动查找是否已存在该模块
			auto it = std::find_if(stats.begin(), stats.end(), [&name](const std::pair<std::string, double> &element) {
				return element.first == name;
			});

			if (it != stats.end()) {
				// 如果存在，累加时间
				it->second += ms;
			} else {
				// 如果不存在，按顺序推入末尾
				stats.push_back({name, ms});
			}
		}

		void print_stats(const std::string &title) {
			if (run_count == 0)
				return;
			std::cout << "\n=== " << title << " Benchmark (Avg of " << run_count << " runs) ===\n";
			std::cout << std::left << std::setw(20) << "Module"
					  << "Time (ms)\n";
			std::cout << "------------------------------\n";

			double sum_of_modules = 0;
			double measured_total = 0;

			// 修改 3: 这里的迭代顺序就是 vector 的存储顺序（即添加顺序）
			for (const auto &item : stats) {
				std::string name = item.first;
				double total_ms = item.second;
				double avg = total_ms / run_count;

				// 如果是 Total 项，单独记录不参与累加
				if (name == " Total Process") {
					measured_total = avg;
					continue;
				}

				sum_of_modules += avg;
				std::cout << std::left << std::setw(20) << name << std::fixed << std::setprecision(3) << avg << "\n";
			}
			std::cout << "------------------------------\n";
			std::cout << std::left << std::setw(20) << "Sum of Modules" << std::fixed << std::setprecision(3)
					  << sum_of_modules << "\n";

			// 最后打印总耗时
			if (measured_total > 0) {
				std::cout << std::left << std::setw(20) << "Measured Total" << measured_total << "\n";
			}
			std::cout << "==============================\n\n";
		}
	};

	Profiler profiler_cpu;
	Profiler profiler_fast;
	Profiler profiler_gpu;

	// 辅助 RAII 计时器，离开作用域自动记录时间
	struct ScopedTimer {
		std::string name;
		Profiler &profiler;
		Timer timer; // 假设你有这个类
		bool active;

		ScopedTimer(std::string n, Profiler &p, bool enable) : name(n), profiler(p), active(enable) {
			if (active)
				timer.start();
		}

		~ScopedTimer() {
			if (active) {
				timer.stop();
				profiler.add(name,
							 timer.elapsed_ms()); // 假设 Timer 有 get_elapsed_ms()
			}
		}
	};
};

#endif // LFISP_H
