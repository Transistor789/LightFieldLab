// =============================================================================
// 1. 必须包含的 CUDA 基础头文件
// =============================================================================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// =============================================================================
// 2. OpenCV CUDA 头文件
// =============================================================================
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>

// [关键修复] 必须包含这个才能使用 StreamAccessor::getStream
#include <opencv2/core/cuda_stream_accessor.hpp>

// [可选] 如果编译器仍抱怨 min/max，可以手动定义 device 版本的 helper
__device__ inline unsigned char dev_min(unsigned char a, unsigned char b) {
	return a < b ? a : b;
}
__device__ inline unsigned char dev_max(unsigned char a, unsigned char b) {
	return a > b ? a : b;
}

__device__ inline unsigned char min4(unsigned char a, unsigned char b,
									 unsigned char c, unsigned char d) {
	// 使用 CUDA 内置的 min 或者我们上面的 dev_min
	return dev_min(dev_min(a, b), dev_min(c, d));
}

__device__ inline unsigned char max4(unsigned char a, unsigned char b,
									 unsigned char c, unsigned char d) {
	return dev_max(dev_max(a, b), dev_max(c, d));
}

// 原地修改 Kernel
__global__ void dpc_kernel_8u_inplace(cv::cuda::PtrStepSz<unsigned char> img,
									  int threshold) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int border = 2;

	// 边界检查
	if (x < border || x >= img.cols - border || y < border
		|| y >= img.rows - border) {
		return;
	}

	// 读取数据 (直接读原图)
	unsigned char center = img(y, x);
	unsigned char val_L = img(y, x - 2);
	unsigned char val_R = img(y, x + 2);
	unsigned char val_U = img(y - 2, x);
	unsigned char val_D = img(y + 2, x);

	unsigned char min_val = min4(val_L, val_R, val_U, val_D);
	unsigned char max_val = max4(val_L, val_R, val_U, val_D);

	// 计算逻辑转 int 防止溢出
	int i_center = (int)center;
	int i_max = (int)max_val;
	int i_min = (int)min_val;

	bool is_hot = (i_center > i_max) && (i_center - i_max > threshold);
	bool is_dead = (i_center < i_min) && (i_min - i_center > threshold);

	// 【关键优化】只有检测到是坏点时，才写入显存！
	if (is_hot || is_dead) {
		int grad_h = abs((int)val_L - (int)val_R);
		int grad_v = abs((int)val_U - (int)val_D);

		unsigned char new_val;
		if (grad_h < grad_v) {
			new_val = (unsigned char)((val_L + val_R) / 2);
		} else if (grad_v < grad_h) {
			new_val = (unsigned char)((val_U + val_D) / 2);
		} else {
			new_val = (unsigned char)((val_L + val_R + val_U + val_D) / 4);
		}

		// 写入回原位置
		img(y, x) = new_val;
	}
}

void launch_dpc_8u_inplace(cv::cuda::GpuMat &img, int threshold,
						   cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	dpc_kernel_8u_inplace<<<grid, block, 0, cuda_stream>>>(img, threshold);
}