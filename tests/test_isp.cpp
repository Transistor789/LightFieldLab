#include "config.h"
#include "lfio.h"
#include "lfisp.h"
#include "utils.h"

#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>
#include <ostream>
#include <vector>

void test_isp() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadLFP("../data/MOD_0015.RAW", meta);
	// lf_img.convertTo(lf_img, CV_8U, 255.0 / 1023.0);
	// wht_img.convertTo(wht_img, CV_8U, 255.0 / 1023.0);

	IspConfig config;
	LFIsp::parseJsonToConfig(j, config);
	LFIsp isp(lf_img, wht_img, config);

	// isp.preview(1.0);
	// cv::imshow("", isp.getPreviewResult());
	// cv::waitKey();

	isp.dpc_fast(DpcMethod::Diretional, 100)
		.blc_fast(66)
		.lsc_awb_fused_fast(0, {})
		.demosaic(config.bayer, DemosaicMethod::Bilinear)
		.ccm_fast_sai({});
	cv::Mat img;
	isp.getResult().convertTo(img, CV_8U, 255.0 / 1023.0);
	cv::imshow("", img);
	cv::waitKey();
}

void test_speed() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadWhiteImageManual("../data/MOD_0015.RAW", meta);

	LFIsp isp;
	IspConfig config;
	isp.parseJsonToConfig(j, config);
	isp.set_lf_img(lf_img).initConfig(wht_img, config);

	Timer timer;
	isp.preview(config);
	// isp.awb_fast({1.0, 1.0, 1.0, 1.0});
	timer.stop();
	timer.print_elapsed_ms();

	// cv::imshow("", isp.getResult());
	// cv::waitKey();
}

void test_demosaic() {
	// 1. 准备数据
	json j, meta;
	std::cout << "[Info] Loading LFP image..." << std::endl;
	// 假设 ReadLFP 返回的是单通道 Raw Bayer 数据 (CV_8U 或 CV_16U)
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);

	if (lf_img.empty()) {
		std::cerr << "[Error] Image load failed!" << std::endl;
		return;
	}
	lf_img.convertTo(lf_img, CV_8U, 255.0 / 1023.0);

	std::cout << "Image Info: " << lf_img.cols << "x" << lf_img.rows
			  << " | Depth: " << (lf_img.depth() == CV_8U ? "8-bit" : "16-bit")
			  << std::endl;

	// 假设 Bayer 格式为 GRBG (你可以根据实际情况修改，如 COLOR_BayerRG2RGB)
	// 注意：Bayer 格式只影响颜色正确性，不影响算法速度
	int bayer_code_linear = cv::COLOR_BayerGB2RGB;
	int bayer_code_ea = cv::COLOR_BayerGB2RGB_EA;
	int bayer_code_vng = cv::COLOR_BayerGB2RGB_VNG;
	int bayer_code_mht = cv::cuda::COLOR_BayerGB2RGB_MHT; // CUDA 独有算法

	cv::Mat res_cpu;
	Timer timer;

	std::cout << "\n>>> Starting CPU Benchmarks..." << std::endl;

	// ---------------------------------------------------------
	// CPU 测试
	// ---------------------------------------------------------

	// 1. CPU Bilinear (最快，质量一般)
	timer.start();
	cv::demosaicing(lf_img, res_cpu, bayer_code_linear);
	timer.stop();
	timer.print_elapsed_ms("CPU: Bilinear");

	// 2. CPU Edge-Aware (EA) (速度快，边缘好，适合实时)
	timer.start();
	cv::demosaicing(lf_img, res_cpu, bayer_code_ea);
	timer.stop();
	timer.print_elapsed_ms("CPU: Edge-Aware (EA)");

	// 3. CPU VNG (Variable Number of Gradients) (最慢，质量最高，适合离线)
	timer.start();
	cv::demosaicing(lf_img, res_cpu, bayer_code_vng);
	timer.stop();
	timer.print_elapsed_ms("CPU: VNG");

	// ---------------------------------------------------------
	// CUDA 测试
	// ---------------------------------------------------------
	int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
	if (cuda_devices > 0) {
		std::cout << "\n>>> Starting CUDA Benchmarks..." << std::endl;
		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

		cv::cuda::GpuMat d_src, d_dst;

		// 4. Upload Time (内存 -> 显存)
		// 这一步在实际 Pipeline 中必不可少，必须计入考量
		timer.start();
		d_src.upload(lf_img);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Upload (Host->Device)");

		// [Warm-up] 预热
		// 第一次调用 CUDA 函数会初始化
		// Context，耗时较长，需要排除以测出真实算法耗时
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_linear);

		// 5. CUDA Bilinear
		timer.start();
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_linear);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Bilinear (Algo only)");

		// 6. CUDA MHT (Malvar-He-Cutler)
		// 这是一个高质量线性插值算法，通常比 CPU VNG 快很多
		timer.start();
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_mht);
		timer.stop();
		timer.print_elapsed_ms("CUDA: MHT (Algo only)");

		// 7. Download Time (显存 -> 内存)
		timer.start();
		d_dst.download(res_cpu);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Download (Device->Host)");

	} else {
		std::cout << "\n[Warning] No CUDA device found. Skipping GPU tests."
				  << std::endl;
	}
	if (cuda_devices > 0) {
		std::cout << "\n>>> Starting CUDA Benchmarks..." << std::endl;
		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

		cv::cuda::GpuMat d_src, d_dst;

		// 4. Upload Time (内存 -> 显存)
		// 这一步在实际 Pipeline 中必不可少，必须计入考量
		timer.start();
		d_src.upload(lf_img);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Upload (Host->Device)");

		// [Warm-up] 预热
		// 第一次调用 CUDA 函数会初始化
		// Context，耗时较长，需要排除以测出真实算法耗时
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_linear);

		// 5. CUDA Bilinear
		timer.start();
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_linear);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Bilinear (Algo only)");

		// 6. CUDA MHT (Malvar-He-Cutler)
		// 这是一个高质量线性插值算法，通常比 CPU VNG 快很多
		timer.start();
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_mht);
		timer.stop();
		timer.print_elapsed_ms("CUDA: MHT (Algo only)");

		// 7. Download Time (显存 -> 内存)
		timer.start();
		d_dst.download(res_cpu);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Download (Device->Host)");

	} else {
		std::cout << "\n[Warning] No CUDA device found. Skipping GPU tests."
				  << std::endl;
	}
	if (cuda_devices > 0) {
		std::cout << "\n>>> Starting CUDA Benchmarks..." << std::endl;
		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

		cv::cuda::GpuMat d_src, d_dst;

		// 4. Upload Time (内存 -> 显存)
		// 这一步在实际 Pipeline 中必不可少，必须计入考量
		timer.start();
		d_src.upload(lf_img);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Upload (Host->Device)");

		// [Warm-up] 预热
		// 第一次调用 CUDA 函数会初始化
		// Context，耗时较长，需要排除以测出真实算法耗时
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_linear);

		// 5. CUDA Bilinear
		timer.start();
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_linear);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Bilinear (Algo only)");

		// 6. CUDA MHT (Malvar-He-Cutler)
		// 这是一个高质量线性插值算法，通常比 CPU VNG 快很多
		timer.start();
		cv::cuda::demosaicing(d_src, d_dst, bayer_code_mht);
		timer.stop();
		timer.print_elapsed_ms("CUDA: MHT (Algo only)");

		// 7. Download Time (显存 -> 内存)
		timer.start();
		d_dst.download(res_cpu);
		timer.stop();
		timer.print_elapsed_ms("CUDA: Download (Device->Host)");

	} else {
		std::cout << "\n[Warning] No CUDA device found. Skipping GPU tests."
				  << std::endl;
	}
}

void test_gpu() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadWhiteImageManual("../data/MOD_0015.RAW", meta);
	IspConfig config;
	LFIsp::parseJsonToConfig(j, config);
	LFIsp isp;
	isp.initConfig(wht_img, config);
	int _;
	LFIO::LoadLookUpTables("../data/calibration/lut_extract_9.bin",
						   isp.maps.extract, _);
	LFIO::LoadLookUpTables("../data/calibration/lut_dehex.bin", isp.maps.dehex,
						   _);
	isp.update_resample_maps();

	config.enableBLC = true;
	config.enableDPC = true;
	config.enableLSC = true;
	config.enableAWB = true;
	config.enableDemosaic = true;
	config.enableCCM = true;
	config.enableGamma = true;
	config.enableExtract = true;
	config.enableDehex = true;
	int count = 5;
	for (int i = 0; i < count; ++i) {
		Timer timer;
		isp.set_lf_gpu(lf_img.clone());
		isp.process_gpu(config);

		timer.stop();
		timer.print_elapsed_ms();
	}
	cv::imwrite("../data/lfp_gpu.png", isp.getResultGpu());
	if (config.enableExtract) {
		auto views_gpu = isp.getSAIsGpu();
		cv::imwrite("../data/center_gpu.png", views_gpu[views_gpu.size() / 2]);
	}

	for (int i = 0; i < count; ++i) {
		Timer timer;

		isp.set_lf_img(lf_img.clone());
		isp.process_fast(config);

		timer.stop();
		timer.print_elapsed_ms();
	}
	cv::imwrite("../data/lfp.png", isp.getResult());
	if (config.enableExtract) {
		auto views = isp.getSAIs();
		cv::imwrite("../data/center.png", views[views.size() / 2]);
	}

	// cv::Mat diff;
	// cv::subtract(isp.getResultGpu(), isp.getResult(), diff);
	// cv::multiply(diff, diff, diff);
	// std::cout << "MSE = " << cv::mean(diff) << std::endl;
	// std::cout << "SSIM = "
	// 		  << cv::quality::QualitySSIM::compute(
	// 				 isp.getResultGpu(), isp.getResult(), cv::noArray())
	// 		  << std::endl;

	for (int i = 0; i < count; ++i) {
		Timer timer;

		isp.set_lf_img(lf_img.clone());
		isp.process(config);

		timer.stop();
		timer.print_elapsed_ms();
	}
	cv::imwrite("../data/lfp_baseline.png", isp.getResult());
	if (config.enableExtract) {
		auto views = isp.getSAIs();
		cv::imwrite("../data/center_baseline.png", views[views.size() / 2]);
	}
}
void benchmark() {
	json j, meta;
	auto lf_img = LFIO::ReadLFP("../data/toy.lfr", j);
	auto wht_img = LFIO::ReadWhiteImageManual("../data/MOD_0015.RAW", meta);
	IspConfig config;
	LFIsp::parseJsonToConfig(j, config);
	LFIsp isp;
	isp.initConfig(wht_img, config);
	int _;
	LFIO::LoadLookUpTables("../data/calibration/lut_extract_9.bin",
						   isp.maps.extract, _);
	LFIO::LoadLookUpTables("../data/calibration/lut_dehex.bin", isp.maps.dehex,
						   _);
	isp.update_resample_maps();

	config.enableBLC = true;
	config.enableDPC = true;
	config.enableLSC = true;
	config.enableAWB = true;
	config.enableDemosaic = true;
	config.enableCCM = true;
	config.enableGamma = true;
	config.enableExtract = true;
	config.enableDehex = true;
	config.benchmark = true;
	int count = 5;

	// ==========================================
	// 测试 GPU Pipeline
	// ==========================================
	std::cout << "Benchmarking GPU Pipeline..." << std::endl;
	isp.profiler_gpu.reset(); // 重置计数器

	// 预热 (Warm-up): 跑一次不计入统计，消除首次 GPU Context
	// 初始化、显存分配的抖动
	isp.set_lf_gpu(lf_img.clone());
	isp.process_gpu(config);
	isp.profiler_gpu.reset();

	for (int i = 0; i < count; ++i) {
		isp.set_lf_gpu(lf_img.clone());
		isp.process_gpu(config);

		isp.profiler_gpu.run_count++;
	}

	isp.profiler_gpu.print_stats("GPU");

	// ==========================================
	// 测试 Fast CPU Pipeline
	// ==========================================
	std::cout << "Benchmarking Fast CPU Pipeline..." << std::endl;
	isp.profiler_fast.reset();

	for (int i = 0; i < count; ++i) {
		isp.set_lf_img(lf_img.clone());
		isp.process_fast(config);
		isp.profiler_fast.run_count++;
	}
	isp.profiler_fast.print_stats("CPU Fast");

	// ==========================================
	// 测试 CPU Pipeline
	// ==========================================
	std::cout << "Benchmarking CPU Pipeline..." << std::endl;
	isp.profiler_cpu.reset();

	for (int i = 0; i < count; ++i) {
		isp.set_lf_img(lf_img.clone());
		isp.process(config);
		isp.profiler_cpu.run_count++;
	}
	isp.profiler_cpu.print_stats("CPU");
}
int main() {
	// test_isp();
	// test_speed();
	// test_demosaic();
	test_gpu();
	benchmark();

	return 0;
}