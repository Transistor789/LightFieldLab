#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

/**
 * @brief 通用线性色彩变换 Kernel (BGR 空间)
 * 公式: P_out = (P_in - mu_s) * T + mu_r
 * 其中 T 是由 CPU 计算好的 3x3 变换矩阵
 */
__global__ void linear_transfer_8uc3_kernel(cv::cuda::PtrStepSz<uchar3> img, float3 mu_s, float3 mu_r, float m0,
											float m1, float m2, float m3, float m4, float m5, float m6, float m7,
											float m8) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < img.cols && y < img.rows) {
		uchar3 p = img(y, x);

		// 1. 去中心化
		float b = (float)p.x - mu_s.x;
		float g = (float)p.y - mu_s.y;
		float r = (float)p.z - mu_s.z;

		// 2. 应用变换矩阵 T (矩阵乘法)
		float nb = m0 * b + m1 * g + m2 * r + mu_r.x;
		float ng = m3 * b + m4 * g + m5 * r + mu_r.y;
		float nr = m6 * b + m7 * g + m8 * r + mu_r.z;

		// 3. 结果钳位并写回
		p.x = (unsigned char)fmaxf(0.0f, fminf(255.0f, nb + 0.5f));
		p.y = (unsigned char)fmaxf(0.0f, fminf(255.0f, ng + 0.5f));
		p.z = (unsigned char)fmaxf(0.0f, fminf(255.0f, nr + 0.5f));

		img(y, x) = p;
	}
}

// Launcher 封装
void launch_linear_transfer_gpu(cv::cuda::GpuMat &img, const float *mu_s, const float *mu_r, const float *T,
								cv::cuda::Stream &stream) {
	cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
	dim3 block(32, 16);
	dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

	linear_transfer_8uc3_kernel<<<grid, block, 0, s>>>(img, make_float3(mu_s[0], mu_s[1], mu_s[2]),
													   make_float3(mu_r[0], mu_r[1], mu_r[2]), T[0], T[1], T[2], T[3],
													   T[4], T[5], T[6], T[7], T[8]);
}