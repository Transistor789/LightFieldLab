#include "trtwrapper.h"

#include <fstream>
#include <iostream>

void TRTLogger::log(Severity severity, const char *msg) noexcept {
	if (severity <= Severity::kWARNING)
		std::cout << "[TRT] " << msg << std::endl;
}

TRTWrapper::TRTWrapper(const std::string &enginePath) {
	// 1. 初始化 Runtime
	runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(logger_));
	if (!runtime_)
		throw std::runtime_error("Failed to create TRT Runtime");

	// 2. 读取 Engine 文件
	std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
	if (!file.good())
		throw std::runtime_error("Engine file not found: " + enginePath);
	size_t size = file.tellg();
	file.seekg(0);
	std::vector<char> buffer(size);
	file.read(buffer.data(), size);

	// 3. 反序列化 Engine
	engine_ = std::unique_ptr<ICudaEngine>(
		runtime_->deserializeCudaEngine(buffer.data(), size));
	if (!engine_)
		throw std::runtime_error("Failed to deserialize Engine");

	// 4. 创建 Context
	context_ =
		std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
	if (!context_)
		throw std::runtime_error("Failed to create Execution Context");

	// === TensorRT 10 新版 API 适配 ===
	// 遍历所有 IO Tensor，自动找到输入和输出的名字及尺寸
	int nbIOTensors = engine_->getNbIOTensors();

	for (int i = 0; i < nbIOTensors; ++i) {
		const char *name = engine_->getIOTensorName(i);
		TensorIOMode mode = engine_->getTensorIOMode(name);
		Dims dims = engine_->getTensorShape(name);

		// 计算该 Tensor 的字节大小
		size_t vol = 1;
		// 注意：TRT 10 的维度可能包含 -1 (动态维度)，这里我们假设是固定维度
		for (int d = 0; d < dims.nbDims; ++d) {
			vol *= (dims.d[d] > 0 ? dims.d[d] : 1);
		}
		size_t sizeBytes = vol * sizeof(float); // 假设 float32

		// 分配显存
		void *ptr = nullptr;
		cudaMalloc(&ptr, sizeBytes);

		if (mode == TensorIOMode::kINPUT) {
			inputName_ = name;
			inputSize_ = sizeBytes;
			inputPtr_ = ptr;
			printf("[TRT] Found Input: %s, Size: %.2f MB\n", name,
				   sizeBytes / 1024.0 / 1024.0);
		} else if (mode == TensorIOMode::kOUTPUT) {
			outputName_ = name;
			outputSize_ = sizeBytes;
			outputPtr_ = ptr;
			printf("[TRT] Found Output: %s, Size: %.2f MB\n", name,
				   sizeBytes / 1024.0 / 1024.0);
		}
	}

	// 创建 CUDA 流
	cudaStreamCreate(&stream_);
}

TRTWrapper::~TRTWrapper() {
	if (inputPtr_)
		cudaFree(inputPtr_);
	if (outputPtr_)
		cudaFree(outputPtr_);
	if (stream_)
		cudaStreamDestroy(stream_);
}

void TRTWrapper::Infer(const float *hostInput, float *hostOutput) {
	// 1. Host -> Device (异步拷贝)
	cudaMemcpyAsync(inputPtr_, hostInput, inputSize_, cudaMemcpyHostToDevice,
					stream_);

	// 2. 绑定地址 (TRT 10 必须步骤)
	// 每次推理前告诉 Context 输入输出在哪里
	context_->setTensorAddress(inputName_.c_str(), inputPtr_);
	context_->setTensorAddress(outputName_.c_str(), outputPtr_);

	// 3. 执行推理 (使用 enqueueV3)
	// enqueueV2 已被移除
	context_->enqueueV3(stream_);

	// 4. Device -> Host (异步拷贝)
	cudaMemcpyAsync(hostOutput, outputPtr_, outputSize_, cudaMemcpyDeviceToHost,
					stream_);

	// 5. 同步等待
	cudaStreamSynchronize(stream_);
}

bool TRTWrapper::setBindingShape(const std::string &tensorName,
								 const std::vector<int> &dims) {
	if (!context_) {
		std::cerr << "[TRT] Context is null!" << std::endl;
		return false;
	}

	// 1. 转换维度格式 std::vector -> nvinfer1::Dims
	nvinfer1::Dims d;
	d.nbDims = (int)dims.size();
	if (d.nbDims > nvinfer1::Dims::MAX_DIMS) {
		std::cerr << "[TRT] Error: Dims too large!" << std::endl;
		return false;
	}
	for (int i = 0; i < d.nbDims; ++i) {
		d.d[i] = dims[i];
	}

	// 2. 调用 TensorRT API 设置形状
	// 注意：TensorRT 8.5+ 和 10.x 推荐使用 setInputShape(name, dims)
	if (!context_->setInputShape(tensorName.c_str(), d)) {
		std::cerr << "[TRT] Failed to set shape for input: " << tensorName
				  << std::endl;
		// 常见失败原因：名字写错、维度范围超出了 build 时的
		// min/maxShapes、或者该 Tensor 不是输入
		return false;
	}

	return true;
}