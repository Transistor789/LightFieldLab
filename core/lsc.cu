#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// 融合算子：8U * 32F Map * Scale -> 8U
__global__ void lsc_8u_apply_32f_kernel(cv::cuda::PtrStepSz<unsigned char> img,
										cv::cuda::PtrStepSz<float> lsc_map,
										float exposure) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	// 1. 读取 8U 像素
	unsigned char val_u8 = img(y, x);

	// 2. 读取 32F 增益
	// 假设 map 和 img 尺寸完全一致
	float gain = lsc_map(y, x);

	// 3. 浮点计算 (exposure 融合在这里乘)
	// 公式: Result = Pixel * Exposure * LSC_Gain
	float res = (float)val_u8 * exposure * gain;

	// 4. 饱和截断 (Saturate Cast)
	if (res > 255.0f)
		res = 255.0f;

	// 5. 写回 8U
	img(y, x) = (unsigned char)(res + 0.5f); // +0.5 四舍五入
}

void launch_lsc_8u_apply_32f(cv::cuda::GpuMat &img,
							 const cv::cuda::GpuMat &lsc_map, float exposure,
							 cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	// 调用 kernel
	lsc_8u_apply_32f_kernel<<<grid, block, 0, cuda_stream>>>(img, lsc_map,
															 exposure);
}