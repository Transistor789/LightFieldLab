#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>


// 1. 修复内核：统一变量名并处理类型转换
__global__ void uvnr_8uc3_guided_kernel(cv::cuda::PtrStepSz<uchar3> img, float inv_sigma_s2,
										float inv_sigma_r2, // 确保与 Launcher 一致
										int radius) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	uchar3 center_pix = img(y, x);
	float center_y = (float)center_pix.x;

	float sum_w = 1.0f;
	float sum_cr = (float)center_pix.y;
	float sum_cb = (float)center_pix.z;

	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			if (i == 0 && j == 0)
				continue;

			int ny = y + i;
			int nx = x + j;

			if (nx >= 0 && nx < img.cols && ny >= 0 && ny < img.rows) {
				uchar3 nb_pix = img(ny, nx);

				float d2 = static_cast<float>(i * i + j * j);
				float r2 =
					static_cast<float>(center_y - (float)nb_pix.x) * static_cast<float>(center_y - (float)nb_pix.x);

				// 修复：使用一致的 inv_sigma_r2 变量名
				float w = expf(-(d2 * inv_sigma_s2 + r2 * inv_sigma_r2));

				sum_w += w;
				sum_cr += w * (float)nb_pix.y;
				sum_cb += w * (float)nb_pix.z;
			}
		}
	}

	center_pix.y = (unsigned char)(sum_cr / sum_w + 0.5f);
	center_pix.z = (unsigned char)(sum_cb / sum_w + 0.5f);

	img(y, x) = center_pix;
}

// 2. 修复 Launcher：确保参数传递正确
void launch_uvnr_8uc3(cv::cuda::GpuMat &img, float sigma_s, float sigma_r, cv::cuda::Stream &stream) {
	// 隐藏 StreamAccessor 细节
	cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

	int radius = static_cast<int>(sigma_s * 1.5f);
	if (radius < 1)
		radius = 1;
	if (radius > 5)
		radius = 5;

	float inv_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
	float inv_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

	dim3 block(32, 16);
	dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

	// 确保调用时变量名对应
	uvnr_8uc3_guided_kernel<<<grid, block, 0, s>>>(img, inv_sigma_s2, inv_sigma_r2, radius);
}