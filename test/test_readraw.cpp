#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <ratio>
#include <vector>

constexpr int LYTRO_WIDTH = 7728;
constexpr int LYTRO_HEIGHT = 5368;

cv::Mat readRawFile(const std::string& filename, int width, int height) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file)
		throw std::runtime_error("Failed to open file");

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<unsigned char> raw10bit(size);
	if (!file.read(reinterpret_cast<char*>(raw10bit.data()), size)) {
		throw std::runtime_error("Failed to read file");
	}

	// 正确创建16-bit矩阵
	cv::Mat raw16bit(LYTRO_HEIGHT, LYTRO_WIDTH, CV_16UC1);
	uint16_t* raw16Ptr = raw16bit.ptr<uint16_t>(); // 使用正确的指针类型

	// 10-bit -> 16-bit转换
	for (int i = 0, index = 0; i < raw10bit.size(); i += 5, index += 4) {
		const uint8_t low_bits = raw10bit[i + 4];
		raw16Ptr[index] = (raw10bit[i] << 2) | ((low_bits >> 6) & 0x03);
		raw16Ptr[index + 1] = (raw10bit[i + 1] << 2) | ((low_bits >> 4) & 0x03);
		raw16Ptr[index + 2] = (raw10bit[i + 2] << 2) | ((low_bits >> 2) & 0x03);
		raw16Ptr[index + 3] = (raw10bit[i + 3] << 2) | (low_bits & 0x03);
	}

	// 16-bit -> 8-bit转换（带缩放）
	cv::Mat raw8bit;
	raw16bit.convertTo(raw8bit, CV_8UC1,
					   255.0 / 1023.0); // 注意分母是4095（12-bit最大值）

	return raw8bit;
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat image =
		readRawFile("../input/MOD_0015.RAW", LYTRO_WIDTH, LYTRO_HEIGHT);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Read " << image.size() << " bytes from file." << std::endl;

	std::cout << "Conversion took "
			  << std::chrono::duration<double, std::milli>(end - start).count()
			  << " ms." << std::endl;
	cv::imwrite("./wht_img.png", image);
	cv::imshow("Image", image);
	cv::waitKey(0);

	// image.convertTo(image, CV_8UC1, 255.0 / 1023.0);

	// std::vector<std::future<void>> futures(4);
	// for (int i = 0; i < 4; i++) {
	// 	futures[i] =
	// 		std::async(std::launch::async, [&raw_16bit, &raw_10bit, i]() {
	// 			for (int j = 0; j < raw_10bit.size() / 5; j++) {
	// 				raw_16bit[j * 4 + i] =
	// 					(raw_10bit[j * 5 + i] << 2)
	// 					| (raw_10bit[j * 5 + 4] >> ((3 - i) * 2)) & 0x03;
	// 			}
	// 		});
	// }

	// for (auto& future : futures) {
	// 	future.wait();
	// }
	// int blockCount = 6;
	// int blockSize = raw_10bit.size() / blockCount;
	// std::vector<std::future<void>> futures(blockCount);

	// for (int i = 0; i < blockCount; ++i) {
	// 	futures[i] = std::async(
	// 		std::launch::async, [&raw_16bit, i, raw_10bit, blockSize]() {
	// 			std::cout << "thread id: " << std::this_thread::get_id()
	// 					  << " processing block " << i << std::endl;
	// 			for (int j = blockSize * i; j < blockSize * (i + 1); j += 5) {
	// 				int index = j * 4 / 5;
	// 				raw_16bit[index] =
	// 					raw_10bit[j] << 2 | (raw_10bit[j + 4] >> 6) & 0x03;
	// 				raw_16bit[index + 1] =
	// 					raw_10bit[j + 1] << 2 | (raw_10bit[j + 4] >> 4) & 0x03;
	// 				raw_16bit[index + 2] =
	// 					raw_10bit[j + 2] << 2 | (raw_10bit[j + 4] >> 2) & 0x03;
	// 				raw_16bit[index + 3] =
	// 					raw_10bit[j + 3] << 2 | (raw_10bit[j + 4] & 0x03);
	// 			}
	// 		});
	// }
	// for (auto& future : futures) {
	// 	future.wait();
	// }
	// for (int i = 0, index = 0; i < raw_10bit.size(); i += 5, index += 4) {
	// 	raw_16bit[index] = raw_10bit[i] << 2 | (raw_10bit[i + 4] >> 6) & 0x03;
	// 	raw_16bit[index + 1] =
	// 		raw_10bit[i + 1] << 2 | (raw_10bit[i + 4] >> 4) & 0x03;
	// 	raw_16bit[index + 2] =
	// 		raw_10bit[i + 2] << 2 | (raw_10bit[i + 4] >> 2) & 0x03;
	// 	raw_16bit[index + 3] = raw_10bit[i + 3] << 2 | raw_10bit[i + 4] & 0x03;
	// }

	return 0;
}