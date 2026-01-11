#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// 辅助：根据坐标奇偶性选择增益
// g00: (偶,偶) | g01: (偶,奇)
// g10: (奇,偶) | g11: (奇,奇)
__global__ void awb_bayer_8u_inplace_kernel(
	cv::cuda::PtrStepSz<unsigned char> img, float g00, float g01, float g10,
	float g11) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	// 读取像素
	unsigned char val = img(y, x);

	// 确定增益
	float gain;
	// 使用位运算判断奇偶 (y&1 等价于 y%2)
	if ((y & 1) == 0) {
		gain = ((x & 1) == 0) ? g00 : g01;
	} else {
		gain = ((x & 1) == 0) ? g10 : g11;
	}

	// 计算并防溢出 (Saturate)
	float res = val * gain;
	if (res > 255.0f)
		res = 255.0f;

	// 写回
	img(y, x) = (unsigned char)(res + 0.5f); // +0.5 做四舍五入
}

// Launcher
void launch_awb_8u(cv::cuda::GpuMat &img, float g00, float g01, float g10,
				   float g11, cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	awb_bayer_8u_inplace_kernel<<<grid, block, 0, cuda_stream>>>(img, g00, g01,
																 g10, g11);
}