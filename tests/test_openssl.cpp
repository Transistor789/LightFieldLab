#include "json.hpp"

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

	return 0;
}
