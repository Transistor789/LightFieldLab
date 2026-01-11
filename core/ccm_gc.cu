#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// 辅助函数：Gamma 计算 (使用 __powf 硬件指令)
__device__ inline float apply_gamma_device_fused(float val, float gamma_val) {
	// 1. 归一化到 0.0 - 1.0
	float norm = val * (1.0f / 255.0f);

	// 2. 幂运算 (注意：这里假设传入的 gamma_val 已经是最终的指数，例如 1/2.2)
	// 使用 __powf 进行快速计算
	float res = __powf(norm, gamma_val);

	// 3. 反归一化
	return res * 255.0f;
}

// =============================================================================
// 融合算子：CCM (3x3 Matrix) + Gamma
// =============================================================================
__global__ void ccm_gamma_fused_kernel(cv::cuda::PtrStepSz<uchar3> img,
									   float c00, float c01, float c02,
									   float c10, float c11, float c12,
									   float c20, float c21, float c22,
									   float gamma_val) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	// 1. 读取 BGR
	uchar3 bgr = img(y, x);

	// 2. 提取 RGB Float
	float r = (float)bgr.z; // z is R
	float g = (float)bgr.y; // y is G
	float b = (float)bgr.x; // x is B

	// 3. CCM 矩阵乘法
	// R' = c00*R + c01*G + c02*B
	float r_new = c00 * r + c01 * g + c02 * b;
	float g_new = c10 * r + c11 * g + c12 * b;
	float b_new = c20 * r + c21 * g + c22 * b;

	// 4. 重要：截断负值 (Clamp to >= 0)
	// 在做 Gamma 之前必须去掉负数，否则 powf 会出 NaN
	r_new = fmaxf(r_new, 0.0f);
	g_new = fmaxf(g_new, 0.0f);
	b_new = fmaxf(b_new, 0.0f);

	// 5. Gamma 校正 (融合点)
	r_new = apply_gamma_device_fused(r_new, gamma_val);
	g_new = apply_gamma_device_fused(g_new, gamma_val);
	b_new = apply_gamma_device_fused(b_new, gamma_val);

	// 6. 最终截断并写回 (Saturate to 0-255)
	// 交换回 BGR 顺序: make_uchar3(B, G, R)
	img(y, x) = make_uchar3((unsigned char)fminf(b_new + 0.5f, 255.0f),
							(unsigned char)fminf(g_new + 0.5f, 255.0f),
							(unsigned char)fminf(r_new + 0.5f, 255.0f));
}

// Launcher
void launch_ccm_gamma_fused(cv::cuda::GpuMat &img, const float *m, float gamma,
							cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	// 直接透传参数
	ccm_gamma_fused_kernel<<<grid, block, 0, cuda_stream>>>(
		img, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], gamma);
}