#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// 辅助函数：对单个 float 值应用 Gamma
__device__ inline float apply_gamma_device(float val, float inv_gamma) {
	// 归一化 (0-255 -> 0.0-1.0) -> pow -> 反归一化
	// fmaxf 确保不计算负数 (虽然 8U 不会是负数)
	return __powf(val * (1.0f / 255.0f), inv_gamma) * 255.0f;
}

// 模板 Kernel：支持 uchar (1通道) 和 uchar3 (3通道)
template <typename T>
__global__ void gc_8u_inplace_kernel(cv::cuda::PtrStepSz<T> img,
									 float inv_gamma) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	T val = img(y, x);

	// 编译期分发：处理单通道或三通道
	// 利用 OpenCV CUDA 的 vec_traits 来判断类型
	if constexpr (sizeof(T) == 1) {
		// === 单通道 (CV_8UC1) ===
		float res = apply_gamma_device((float)val, inv_gamma);
		img(y, x) = (T)(res + 0.5f);
	} else {
		// === 三通道 (CV_8UC3) ===
		// 显式转为 uchar3 处理
		uchar3 v = *(uchar3 *)&val;

		float b = apply_gamma_device((float)v.x, inv_gamma);
		float g = apply_gamma_device((float)v.y, inv_gamma);
		float r = apply_gamma_device((float)v.z, inv_gamma);

		// 写回
		uchar3 res;
		res.x = (unsigned char)(b + 0.5f);
		res.y = (unsigned char)(g + 0.5f);
		res.z = (unsigned char)(r + 0.5f);
		img(y, x) = *(T *)&res;
	}
}

// Launcher
void launch_gc_8u(cv::cuda::GpuMat &img, float gamma,
				  cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	// 预计算 1/gamma，减少 kernel 内部除法
	// float inv_gamma = 1.0f / gamma;

	if (img.type() == CV_8UC1) {
		gc_8u_inplace_kernel<unsigned char>
			<<<grid, block, 0, cuda_stream>>>(img, gamma);
	} else if (img.type() == CV_8UC3) {
		gc_8u_inplace_kernel<uchar3>
			<<<grid, block, 0, cuda_stream>>>(img, gamma);
	}
}