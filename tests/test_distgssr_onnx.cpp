#include "utils.h"

#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

// ==========================================
// 1. 配置参数 & 跨平台路径处理
// ==========================================

// 【修复点】：根据操作系统定义模型路径类型
#ifdef _WIN32
// Windows: 使用宽字符
const std::wstring MODEL_PATH = L"../models/DistgSSR_2x_5x5.onnx";
// using ORTCHAR_T = wchar_t;
#else
// Linux: 使用标准字符
const std::string MODEL_PATH = "../models/DistgSSR_2x_5x5.onnx";
// using ORTCHAR_T = char;
#endif

// 数据路径
const std::string SRC_IMG_DIR = "../data/INRIA_Lytro_Hublais__Decoded/";
const std::string SRC_IMG_PREFIX = "view";

// 维度参数
const int SOURCE_GRID_SIZE = 9; // 原始数据 9x9
const int MODEL_ANG_RES = 5;	// 模型需要 5x5
const int SCALE = 2;			// 超分倍率

// 分块参数
const int PATCH_SIZE = 196;
const int PADDING = 8;

// ==========================================
// 推理后端选择开关
// ==========================================
// true  = 使用 TensorRT (第一次启动慢，后续极快，需 fp16 支持)
// false = 使用 CUDA (启动快，速度稳定)
const bool USE_TENSORRT =
	false; // 如果你在 Linux 且没配好 TRT 环境变量，建议先设为 false 测试 CUDA

// ==========================================
// 2. 推理引擎类
// ==========================================
class DistgSSRInferencer {
public:
	DistgSSRInferencer(Ort::Session &session) : session_(session) {}

	cv::Mat Infer(const std::vector<cv::Mat> &yViews) {
		// 校验输入
		if (yViews.size() != MODEL_ANG_RES * MODEL_ANG_RES) {
			std::cerr << "Error: Expected 25 views, got " << yViews.size()
					  << std::endl;
			return cv::Mat();
		}

		int H = yViews[0].rows;
		int W = yViews[0].cols;
		int outH = H * SCALE;
		int outW = W * SCALE;

		cv::Mat resultCenter = cv::Mat::zeros(outH, outW, CV_8UC1);
		int step = PATCH_SIZE - 2 * PADDING;

		printf("Starting tiled inference on discrete views...\n");
		printf("Source Size: %dx%d. Patch: %d. Padding: %d\n", H, W, PATCH_SIZE,
			   PADDING);

		// === 空间循环 ===
		for (int y = 0; y < H; y += step) {
			for (int x = 0; x < W; x += step) {
				// 1. 计算 Patch ROI
				int x_start = std::max(0, x - PADDING);
				int y_start = std::max(0, y - PADDING);
				int x_end = std::min(W, x + step + PADDING);
				int y_end = std::min(H, y + step + PADDING);

				int ph = y_end - y_start;
				int pw = x_end - x_start;

				// 2. 构造 Tensor
				int miniMosaicH = ph * MODEL_ANG_RES;
				int miniMosaicW = pw * MODEL_ANG_RES;

				std::vector<int64_t> inputShape = {1, 1, miniMosaicH,
												   miniMosaicW};
				size_t inputLen = miniMosaicH * miniMosaicW;
				std::vector<float> inputData(inputLen);
				float *tensorBasePtr = inputData.data();

				// 填数据
				for (int u = 0; u < MODEL_ANG_RES; ++u) {
					for (int v = 0; v < MODEL_ANG_RES; ++v) {
						int viewIdx = u * MODEL_ANG_RES + v;
						cv::Mat patch =
							yViews[viewIdx](cv::Rect(x_start, y_start, pw, ph));

						for (int r = 0; r < ph; ++r) {
							int globalRow = u * ph + r;
							int globalCol = v * pw;
							float *dstPtr = tensorBasePtr
											+ globalRow * miniMosaicW
											+ globalCol;
							memcpy(dstPtr, patch.ptr<float>(r),
								   pw * sizeof(float));
						}
					}
				}

				// 3. 推理
				Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
					OrtArenaAllocator, OrtMemTypeDefault);
				Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
					memoryInfo, inputData.data(), inputData.size(),
					inputShape.data(), inputShape.size());

				const char *inputNames[] = {"input"};
				const char *outputNames[] = {"output"};

				auto outputTensors =
					session_.Run(Ort::RunOptions{nullptr}, inputNames,
								 &inputTensor, 1, outputNames, 1);

				// 4. 后处理
				float *outPtr = outputTensors[0].GetTensorMutableData<float>();
				int outPh = ph * SCALE;
				int outPw = pw * SCALE;
				int outMiniH = miniMosaicH * SCALE;
				int outMiniW = miniMosaicW * SCALE;

				cv::Mat outMiniMat(outMiniH, outMiniW, CV_32F, outPtr);

				int center_u = MODEL_ANG_RES / 2;
				int center_v = MODEL_ANG_RES / 2;
				int roiY = center_u * outPh;
				int roiX = center_v * outPw;

				cv::Mat srPatch =
					outMiniMat(cv::Rect(roiX, roiY, outPw, outPh));
				cv::Mat srPatch8U;
				srPatch.convertTo(srPatch8U, CV_8U, 255.0);

				// 5. 拼接
				int valid_x_src = (x == 0) ? 0 : PADDING * SCALE;
				int valid_y_src = (y == 0) ? 0 : PADDING * SCALE;
				int valid_w = std::min(step, W - x) * SCALE;
				int valid_h = std::min(step, H - y) * SCALE;

				int dst_x = x * SCALE;
				int dst_y = y * SCALE;

				srPatch8U(cv::Rect(valid_x_src, valid_y_src, valid_w, valid_h))
					.copyTo(
						resultCenter(cv::Rect(dst_x, dst_y, valid_w, valid_h)));
			}
		}
		printf("Inference finished.\n");
		return resultCenter;
	}

private:
	Ort::Session &session_;
};

// ==========================================
// 3. Main 函数
// ==========================================
int main() {
	// ------------------------------------------
	// A. 初始化 ONNX Runtime (根据参数选择后端)
	// ------------------------------------------
	std::string logId = USE_TENSORRT ? "DistgSSR_TRT" : "DistgSSR_CUDA";
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, logId.c_str());
	Ort::SessionOptions sessionOptions;

	// 通用优化选项
	sessionOptions.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_ALL);

	try {
		if (USE_TENSORRT) {
			// ==========================================
			// 分支 1: TensorRT (高性能模式)
			// ==========================================
			printf("[Init] Attempting to enable TensorRT Provider...\n");

			OrtTensorRTProviderOptions trt_options{};
			trt_options.device_id = 0;

			// FP16 加速
			trt_options.trt_fp16_enable = 1;

			// Engine 缓存
			trt_options.trt_engine_cache_enable = 1;
			trt_options.trt_engine_cache_path = "./trt_cache";

			// 显存限制 (4GB)
			trt_options.trt_max_workspace_size = 4L * 1024 * 1024 * 1024;

			sessionOptions.AppendExecutionProvider_TensorRT(trt_options);

			// 添加 CUDA 作为回退
			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = 0;
			sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

			printf("[Info] TensorRT enabled (with FP16 & Cache).\n");
		} else {
			// ==========================================
			// 分支 2: 纯 CUDA (稳定模式)
			// ==========================================
			printf("[Init] Attempting to enable CUDA Provider...\n");

			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = 0;
			sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
			printf("[Info] CUDA enabled.\n");
		}
	} catch (const std::exception &e) {
		std::cerr << "[Error] Provider configuration failed: " << e.what()
				  << std::endl;
		std::cerr << "[Info] Falling back to CPU.\n";
	}

	printf("Creating session...\n");
	if (USE_TENSORRT) {
		printf(
			"Note: If using TRT for the first time or input size changed, this "
			"may take a few minutes.\n");
	}

	// 【修复点】：使用 .c_str()，它会根据 MODEL_PATH 的类型返回 char* (Linux)
	// 或 wchar_t* (Windows)
	//  Ort::Session 构造函数会自动重载匹配正确的类型
	Ort::Session session(env, MODEL_PATH.c_str(), sessionOptions);
	printf("Session created successfully.\n");

	// ------------------------------------------
	// B. 读取离散的 25 张图像
	// ------------------------------------------
	std::vector<cv::Mat> y_views;
	cv::Mat centerCb, centerCr;
	int H = 0, W = 0;
	int start_idx = (SOURCE_GRID_SIZE - MODEL_ANG_RES) / 2;

	printf("Loading 25 discrete views from %s ...\n", SRC_IMG_DIR.c_str());

	for (int u = 0; u < MODEL_ANG_RES; ++u) {
		for (int v = 0; v < MODEL_ANG_RES; ++v) {
			int src_u = start_idx + u;
			int src_v = start_idx + v;

			// 构造文件名
			char filename[256];
			// 注意：这里请确保你的文件名格式与实际图片一致
			sprintf(filename, "%s%s_%02d_%02d.png", SRC_IMG_DIR.c_str(),
					SRC_IMG_PREFIX.c_str(), u + 1, v + 1);

			cv::Mat img = cv::imread(filename);
			if (img.empty()) {
				std::cerr << "[Error] Load failed: " << filename << std::endl;
				return -1;
			}
			if (H == 0) {
				H = img.rows;
				W = img.cols;
			}

			cv::Mat ycrcb;
			cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
			std::vector<cv::Mat> chans;
			cv::split(ycrcb, chans);

			cv::Mat yFloat;
			chans[0].convertTo(yFloat, CV_32F, 1.0f / 255.0f);
			y_views.push_back(yFloat);

			if (u == MODEL_ANG_RES / 2 && v == MODEL_ANG_RES / 2) {
				centerCr = chans[1].clone();
				centerCb = chans[2].clone();
			}
		}
	}
	printf("Loaded 25 views. Size: %dx%d.\n", H, W);

	// ------------------------------------------
	// C. 执行推理
	// ------------------------------------------
	DistgSSRInferencer inferencer(session);
	Timer timer;
	cv::Mat srY = inferencer.Infer(y_views);
	timer.stop();
	printf("Inference time: ");
	timer.print_elapsed_ms();

	// ------------------------------------------
	// D. 后处理
	// ------------------------------------------
	printf("Post-processing...\n");
	int outH = srY.rows;
	int outW = srY.cols;

	cv::Mat srCr, srCb;
	cv::resize(centerCr, srCr, cv::Size(outW, outH), 0, 0, cv::INTER_CUBIC);
	cv::resize(centerCb, srCb, cv::Size(outW, outH), 0, 0, cv::INTER_CUBIC);

	std::vector<cv::Mat> outChans = {srY, srCr, srCb};
	cv::Mat srYCrCb, srBGR;
	cv::merge(outChans, srYCrCb);
	cv::cvtColor(srYCrCb, srBGR, cv::COLOR_YCrCb2BGR);

	// 根据模式修改文件名
	std::string suffix = USE_TENSORRT ? "_TRT" : "_CUDA";
	std::string savePath = "result_DistgSSR" + suffix + ".png";

	cv::imwrite(savePath, srBGR);
	printf("Saved result to: %s\n", savePath.c_str());

	return 0;
}