#ifndef TRTWRAPPER_H
#define TRTWRAPPER_H

#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

// 日志记录器
class TRTLogger : public ILogger {
	void log(Severity severity, const char *msg) noexcept override;
};

class TRTWrapper {
public:
	TRTWrapper(const std::string &enginePath);

	~TRTWrapper();

	// 核心推理函数 (适配 TRT 10)
	void Infer(const float *hostInput, float *hostOutput);

	/**
	 * @brief [新增] 设置输入 Tensor 的动态形状 (TensorRT 10.x)
	 * @param tensorName Tensor 的名字 (导出 ONNX 时设置的名字，如 "input")
	 * @param dims 维度列表，例如 {1, 1, 576, 576}
	 * @return true 设置成功
	 */
	bool setBindingShape(const std::string &tensorName,
						 const std::vector<int> &dims);

private:
	TRTLogger logger_;
	std::unique_ptr<IRuntime> runtime_;
	std::unique_ptr<ICudaEngine> engine_;
	std::unique_ptr<IExecutionContext> context_;

	// 保存名字和指针
	std::string inputName_;
	std::string outputName_;

	void *inputPtr_ = nullptr;
	void *outputPtr_ = nullptr;

	size_t inputSize_ = 0;
	size_t outputSize_ = 0;

	cudaStream_t stream_ = nullptr;
};

#endif