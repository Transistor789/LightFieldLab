#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// 辅助函数：对单个 float 值应用 sRGB 分段式 Gamma
__device__ inline float apply_gamma_device(float val, float inv_gamma) {
	// 1. 归一化 (0-255 -> 0.0-1.0)
	float norm = val * (1.0f / 255.0f);
	float res;

	// 2. sRGB 标准分段逻辑参数
	const float low_threshold = 0.0031308f;
	const float low_slope = 12.92f;
	const float alpha = 0.055f;

	// 3. 执行分段计算
	if (norm <= low_threshold) {
		// 暗部线性拉伸，抑制噪声放大
		res = low_slope * norm;
	} else {
		// 亮部非线性压缩，使用 sRGB 标准偏移补偿
		res = (1.0f + alpha) * __powf(norm, inv_gamma) - alpha;
	}

	// 4. 反归一化并钳位 (确保不溢出 0-255)
	return fmaxf(0.0f, fminf(255.0f, res * 255.0f));
}

// 模板 Kernel：支持 uchar (1通道) 和 uchar3 (3通道)
template <typename T>
__global__ void gc_8u_inplace_kernel(cv::cuda::PtrStepSz<T> img, float inv_gamma) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	T val = img(y, x);

	if constexpr (sizeof(T) == 1) {
		// === 单通道 (CV_8UC1) ===
		float res = apply_gamma_device((float)val, inv_gamma);
		img(y, x) = (T)(res + 0.5f);
	} else {
		// === 三通道 (CV_8UC3) ===
		uchar3 v = *(uchar3 *)&val;

		float b = apply_gamma_device((float)v.x, inv_gamma);
		float g = apply_gamma_device((float)v.y, inv_gamma);
		float r = apply_gamma_device((float)v.z, inv_gamma);

		uchar3 res;
		res.x = (unsigned char)(b + 0.5f);
		res.y = (unsigned char)(g + 0.5f);
		res.z = (unsigned char)(r + 0.5f);
		img(y, x) = *(T *)&res;
	}
}

// Launcher
void launch_gc_8u(cv::cuda::GpuMat &img, float gamma, cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x), cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	// 如果未指定 gamma，默认使用 1/2.4 ≈ 0.416667f
	float inv_gamma = (gamma > 0) ? gamma : 0.416667f;

	if (img.type() == CV_8UC1) {
		gc_8u_inplace_kernel<unsigned char><<<grid, block, 0, cuda_stream>>>(img, inv_gamma);
	} else if (img.type() == CV_8UC3) {
		gc_8u_inplace_kernel<uchar3><<<grid, block, 0, cuda_stream>>>(img, inv_gamma);
	}
}