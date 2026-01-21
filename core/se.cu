#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 必须包含这个才能看到 GpuMat 的成员函数（如 .cols, .ptr）
#include <opencv2/core/cuda.hpp>
// 必须包含这个才能使用 cv::cuda::StreamAccessor
#include <opencv2/core/cuda_stream_accessor.hpp>
// 必须包含这个才能让 GpuMat 自动转换为 PtrStepSz
#include <opencv2/core/cuda/common.hpp>

// 手写 Kernel：一次读取，原地修改，一次写回
__global__ void saturation_kernel(cv::cuda::PtrStepSz<uchar3> img, float factor) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < img.cols && y < img.rows) {
		uchar3 p = img(y, x); // 一次性读取 Y, Cr, Cb

		// p.x 是 Y (不处理), p.y 是 Cr, p.z 是 Cb
		// 使用 __saturate_cast 或手动 clamp 保证不溢出
		float cr = (static_cast<float>(p.y) - 128.0f) * factor + 128.0f;
		float cb = (static_cast<float>(p.z) - 128.0f) * factor + 128.0f;

		p.y = (unsigned char)fmaxf(0.0f, fminf(255.0f, cr + 0.5f));
		p.z = (unsigned char)fmaxf(0.0f, fminf(255.0f, cb + 0.5f));

		img(y, x) = p; // 写回显存
	}
}

void launch_se_gpu(cv::cuda::GpuMat &img, float factor, cv::cuda::Stream &stream) {
	// 将提取逻辑隐藏在 Launcher 内部
	cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

	dim3 block(32, 16);
	dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

	// 启动手写 Kernel
	saturation_kernel<<<grid, block, 0, s>>>(img, factor);
}