#include "../include/json.hpp"
#include "../include/lfload.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openssl/sha.h>

using json = nlohmann::json;

int main() {
	const char *msg = "hello";
	unsigned char hash[SHA_DIGEST_LENGTH];
	SHA1(reinterpret_cast<const unsigned char *>(msg), strlen(msg), hash);

	std::cout << "SHA1 of 'hello' is: ";
	for (unsigned char i : hash) printf("%02x", i);
	std::cout << std::endl;

	std::filesystem::path cwd = std::filesystem::current_path();
	std::cout << "Current working directory: " << cwd << std::endl;

	cv::Mat lytro_resized, png_resized, raw_resized;

	auto lytro = LFLoad::loadImageFile("../input/toy.lfr");
	std::cout << "toy.lfr size" << lytro.size() << std::endl;
	cv::resize(lytro, lytro_resized, cv::Size(), 0.1, 0.1, cv::INTER_LINEAR);
	cv::imshow("lytro_resized", lytro_resized);

	// auto png = LFLoad::loadImageFile("./output.png");
	// std::cout << "output.png size" << png.size() << std::endl;
	// cv::resize(png, png_resized, cv::Size(), 0.1, 0.1, cv::INTER_LINEAR);
	// cv::imshow("png_resized", png_resized);

	auto raw = LFLoad::loadImageFile("../input/MOD_0015.RAW");
	std::cout << "MOD_0015.RAW size" << raw.size() << std::endl;
	cv::resize(raw, raw_resized, cv::Size(), 0.1, 0.1, cv::INTER_LINEAR);
	cv::imshow("raw_resized", raw_resized);

	cv::waitKey();

	return 0;
}
