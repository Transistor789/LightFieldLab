#include <chrono>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

// 使用 cv::Mat 进行性能测试
void test_performance_mat(const cv::Mat &input, const std::string &mode) {
	cv::Mat src, dst;
	input.copyTo(src);

	cv::setNumThreads(0); // 恢复默认线程数

	auto start = std::chrono::high_resolution_clock::now();

	// 改用计算密集型操作（如Canny边缘检测）
	for (int i = 0; i < 50; i++) { // 减少迭代次数但增加单次计算量
		cv::Canny(src, dst, 50, 150);
		// 或使用: cv::dft(src, dst, cv::DFT_COMPLEX_OUTPUT);
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
			.count();
	std::cout << mode << " time: " << duration << " ms" << std::endl;
}

// 使用 cv::UMat 进行性能测试
void test_performance_umat(bool use_opencl, const cv::UMat &input,
						   const std::string &mode) {
	cv::UMat src, dst;
	input.copyTo(src);

	// 配置后端
	cv::ocl::setUseOpenCL(use_opencl);

	if (!use_opencl) {
		cv::setNumThreads(0); // 恢复默认线程数
	}

	auto start = std::chrono::high_resolution_clock::now();

	// 改用计算密集型操作（如Canny边缘检测）
	for (int i = 0; i < 50; i++) { // 减少迭代次数但增加单次计算量
		cv::Canny(src, dst, 50, 150);
		// 或使用: cv::dft(src, dst, cv::DFT_COMPLEX_OUTPUT);
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
			.count();

	std::cout << mode << " time: " << duration << " ms" << std::endl;
}

int main(int argc, char **argv) {
	std::cout << "Testing OpenCL performance" << std::endl;
	cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Error: The image is empty." << std::endl;
		return -1;
	}

	// 正式测试 (cv::Mat)
	test_performance_mat(img, "Mat");

	// 使用 UMat 测试
	cv::UMat umat_img;
	img.copyTo(umat_img);
	test_performance_umat(true, umat_img, "UMat OpenCL Enable");
	test_performance_umat(false, umat_img, "UMat OpenCL Disable");

	return 0;
}
