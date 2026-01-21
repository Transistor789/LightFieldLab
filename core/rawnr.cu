#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

/**
 * @brief Bayer RAW 域分通道双边滤波核函数
 * @param src 输入图像
 * @param dst 输出图像
 * @param inv_2sigma_s2 空间域权重系数 1/(2*sigma_s^2)
 * @param inv_2sigma_r2 值域权重系数 1/(2*sigma_r^2)
 * @param radius 搜索半径（针对同色通道，实际像素跨度为 2*radius）
 */
__global__ void nr_bayer_8u_kernel(cv::cuda::PtrStepSz<unsigned char> src, cv::cuda::PtrStepSz<unsigned char> dst,
								   float inv_2sigma_s2, float inv_2sigma_r2, int radius) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= src.cols || y >= src.rows)
		return;

	unsigned char center_val = src(y, x);
	float sum_w = 1.0f;
	float sum_i = (float)center_val;

	// 遍历邻域，步长为 2 确保采样同色像素 (R-R, G-G, B-B)
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			if (i == 0 && j == 0)
				continue;

			int ny = y + i * 2;
			int nx = x + j * 2;

			// 边界检查
			if (nx >= 0 && nx < src.cols && ny >= 0 && ny < src.rows) {
				unsigned char neighbor_val = src(ny, nx);

				// 计算空间距离平方 (dist^2) 和亮度距离平方 (range^2)
				float d2 = (float)(i * i + j * j) * 4.0f;
				float r2 = (float)(center_val - neighbor_val) * (center_val - neighbor_val);

				// 双边权重计算: W = exp(-d^2/(2*s^2) - r^2/(2*r^2))
				float w = __expf(-(d2 * inv_2sigma_s2 + r2 * inv_2sigma_r2));

				sum_w += w;
				sum_i += w * (float)neighbor_val;
			}
		}
	}

	// 归一化并写回
	dst(y, x) = (unsigned char)(sum_i / sum_w + 0.5f);
}

/**
 * @brief Launcher: 启动 RAW 域降噪
 * @param src 输入 GpuMat
 * @param dst 输出 GpuMat (建议与 src 尺寸相同)
 * @param sigma_s 空间域标准差（建议 1.0 - 2.0）
 * @param sigma_r 值域标准差（建议 5.0 - 15.0）
 */
void launch_nr_8u(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float sigma_s, float sigma_r,
				  cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(src.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));

	// 预计算常数减少 Kernel 内计算量
	float inv_2sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
	float inv_2sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

	// 自动确定搜索半径 (3 * sigma_s)
	int radius = static_cast<int>(sigma_s * 1.5f);
	if (radius < 1)
		radius = 1;

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	nr_bayer_8u_kernel<<<grid, block, 0, cuda_stream>>>(src, dst, inv_2sigma_s2, inv_2sigma_r2, radius);
}