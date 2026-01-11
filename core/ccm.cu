#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>


// 使用常量内存或寄存器传参均可，这里为了简单直接传参
// 矩阵通常是 3x3
__global__ void ccm_8uc3_inplace_kernel(cv::cuda::PtrStepSz<uchar3> img,
										float c00, float c01, float c02,
										float c10, float c11, float c12,
										float c20, float c21, float c22) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	// 1. 读取 BGR (OpenCV 默认顺序)
	uchar3 bgr = img(y, x);

	// 2. 转为 RGB Float (方便矩阵乘法)
	float r = (float)bgr.z; // z is R
	float g = (float)bgr.y; // y is G
	float b = (float)bgr.x; // x is B

	// 3. 矩阵乘法 (Result = CCM * Source)
	// R' = c00*R + c01*G + c02*B
	float r_new = c00 * r + c01 * g + c02 * b;
	float g_new = c10 * r + c11 * g + c12 * b;
	float b_new = c20 * r + c21 * g + c22 * b;

	// 4. 饱和截断 (Saturate)
	// CUDA 内置 fminf/fmaxf 指令非常快
	r_new = fminf(fmaxf(r_new, 0.0f), 255.0f);
	g_new = fminf(fmaxf(g_new, 0.0f), 255.0f);
	b_new = fminf(fmaxf(b_new, 0.0f), 255.0f);

	// 5. 写回 BGR (交换顺序)
	// make_uchar3(B, G, R)
	img(y, x) = make_uchar3((unsigned char)(b_new + 0.5f),
							(unsigned char)(g_new + 0.5f),
							(unsigned char)(r_new + 0.5f));
}

// Launcher
void launch_ccm_8uc3(cv::cuda::GpuMat &img, const float *m,
					 cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	// 直接把数组展开传进去，避免在 kernel 里访问 global memory 的指针
	ccm_8uc3_inplace_kernel<<<grid, block, 0, cuda_stream>>>(
		img, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
}