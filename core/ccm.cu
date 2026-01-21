#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// 手写 CCM Kernel：[R, G, B]^T_out = CCM * [R, G, B]^T_in
__global__ void ccm_8uc3_kernel(cv::cuda::PtrStepSz<uchar3> img, float m0, float m1, float m2, float m3, float m4,
								float m5, float m6, float m7, float m8) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < img.cols && y < img.rows) {
		uchar3 p = img(y, x);

		// 使用传入的参数进行计算 (此时 m0-m8 位于寄存器/常量内存中，速度极快)
		float r = m0 * p.z + m1 * p.y + m2 * p.x;
		float g = m3 * p.z + m4 * p.y + m5 * p.x;
		float b = m6 * p.z + m7 * p.y + m8 * p.x;

		p.z = (unsigned char)fmaxf(0.0f, fminf(255.0f, r + 0.5f));
		p.y = (unsigned char)fmaxf(0.0f, fminf(255.0f, g + 0.5f));
		p.x = (unsigned char)fmaxf(0.0f, fminf(255.0f, b + 0.5f));

		img(y, x) = p;
	}
}

void launch_ccm_8uc3(cv::cuda::GpuMat &img, const float *h_ccm, cv::cuda::Stream &stream) {
	cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

	dim3 block(32, 16);
	dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

	// 【关键修复】将主机指针指向的内容作为数值传入，而不是传指针
	ccm_8uc3_kernel<<<grid, block, 0, s>>>(img, h_ccm[0], h_ccm[1], h_ccm[2], h_ccm[3], h_ccm[4], h_ccm[5], h_ccm[6],
										   h_ccm[7], h_ccm[8]);
}