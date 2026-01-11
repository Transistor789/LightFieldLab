#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// =============================================================================
// 融合算子：LSC + AWB + Exposure 一步到位
// =============================================================================
// 输入：8U Raw 图
// 输入：32F LSC Map
// 输入：Exposure 标量
// 输入：AWB 4通道增益 (g00, g01, g10, g11)
// 输出：原地修改 img
__global__ void lsc_awb_fused_kernel(cv::cuda::PtrStepSz<unsigned char> img,
									 cv::cuda::PtrStepSz<float> lsc_map,
									 float exposure, float g00, float g01,
									 float g10, float g11) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img.cols || y >= img.rows)
		return;

	// 1. 读取 8U 像素 (Global Memory Read 1)
	unsigned char val_u8 = img(y, x);

	// 2. 读取 LSC 增益 (Global Memory Read 2)
	float lsc_gain = lsc_map(y, x);

	// 3. 确定 AWB 增益 (寄存器操作)
	float awb_gain;
	// 使用位运算判断奇偶性 (比 % 快)
	if ((y & 1) == 0) {
		awb_gain = ((x & 1) == 0) ? g00 : g01;
	} else {
		awb_gain = ((x & 1) == 0) ? g10 : g11;
	}

	// 4. 融合计算
	// 公式：Result = Pixel * (Exposure * LSC * AWB)
	// 所有的乘法都在寄存器内完成，极快
	float total_gain = exposure * lsc_gain * awb_gain;
	float res = (float)val_u8 * total_gain;

	// 5. 饱和截断 (Saturate Cast)
	if (res > 255.0f)
		res = 255.0f;

	// 6. 写回 8U (Global Memory Write 1)
	img(y, x) = (unsigned char)(res + 0.5f);
}

// Launcher
void launch_fused_lsc_awb(cv::cuda::GpuMat &img,
						  const cv::cuda::GpuMat &lsc_map, float exposure,
						  float g00, float g01, float g10, float g11,
						  cv::cuda::Stream &stream) {
	dim3 block(32, 16);
	dim3 grid(cv::cuda::device::divUp(img.cols, block.x),
			  cv::cuda::device::divUp(img.rows, block.y));

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

	lsc_awb_fused_kernel<<<grid, block, 0, cuda_stream>>>(
		img, lsc_map, exposure, g00, g01, g10, g11);
}