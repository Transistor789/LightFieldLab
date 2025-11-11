#include "lfload.h"

#include <filesystem>
#include <fstream>
#include <future>
#include <opencv2/opencv.hpp>
#include <openssl/sha.h>
#include <string>

std::vector<uint8_t> LFLoad::readRawFile(const std::string &filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file) {
		throw std::runtime_error("readRawFile: file not exist!");
	}
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer(size);
	if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
		throw std::runtime_error("readRawFile: read error!");
	}
	return buffer;
}
cv::Mat LFLoad::convertRaw8ToMat(uint8_t *src, int width, int height,
								 bool isFloat) {
	if (src == nullptr) {
		throw std::runtime_error("convertRaw8ToMat: src is nullptr!");
	}
	auto result = cv::Mat(height, width, CV_8UC1, src);
	if (isFloat) {
		result.convertTo(result, CV_32FC1, 1.0 / 255.0);
	}
	return result;
}
cv::Mat LFLoad::convertRaw10ToMat(const uint8_t *src, int width, int height,
								  bool isFloat) {
	if (src == nullptr) {
		throw std::runtime_error("convertRaw10ToMat: src is nullptr!");
	}
	auto mat16 = cv::Mat(height, width, CV_16UC1);
	auto *pTemp = mat16.ptr<uint16_t>();
	int dataSize = width * height * 5 / 4;
	for (int i = 0, index = 0; i < dataSize; i += 5, index += 4) {
		const uint8_t low_bits = src[i + 4];
		pTemp[index] = (src[i] << 2) | ((low_bits >> 6) & 0x03);
		pTemp[index + 1] = (src[i + 1] << 2) | ((low_bits >> 4) & 0x03);
		pTemp[index + 2] = (src[i + 2] << 2) | ((low_bits >> 2) & 0x03);
		pTemp[index + 3] = (src[i + 3] << 2) | (low_bits & 0x03);
	}
	cv::Mat result;
	if (isFloat) {
		mat16.convertTo(result, CV_32FC1, 1.0 / 1023.0);
	} else {
		mat16.convertTo(result, CV_8UC1, 255.0 / 1023.0);
	}
	return result;
}
cv::Mat LFLoad::gammaConvert(const cv::Mat &src, bool inverse) {
	if (src.empty()) {
		throw std::runtime_error("gammaConvert: src is empty!");
	}

	// 转换为 float32 格式，归一化到 [0, 1]
	cv::Mat srcFloat;
	if (src.depth() != CV_32F) {
		src.convertTo(srcFloat, CV_32F, 1.0 / 255.0);
	} else {
		srcFloat = src;
	}

	std::vector<cv::Mat> channels;
	cv::split(srcFloat, channels); // 拆分通道

	constexpr float A = 0.055f;
	constexpr float ALPHA = 1.055f;

	for (auto &ch : channels) {
		if (!inverse) {
			// 线性 -> sRGB
			constexpr float BETA = 0.0031308f;
			cv::Mat mask = ch >= BETA;

			cv::Mat gammaPart, linearPart;
			cv::pow(ch, 1.0f / 2.4f, gammaPart);
			gammaPart = ALPHA * gammaPart - A;

			linearPart = ch * 12.92f;
			gammaPart.copyTo(ch, mask);	  // 用 gamma 区覆盖
			linearPart.copyTo(ch, ~mask); // 用线性区覆盖其余部分
		} else {
			// sRGB -> 线性
			constexpr float THRESHOLD = 0.04045f;
			cv::Mat mask = ch > THRESHOLD;

			cv::Mat srcAdjusted = (ch + A) / ALPHA;
			cv::Mat gammaPart;
			cv::pow(srcAdjusted, 2.4f, gammaPart);

			cv::Mat linearPart = ch / 12.92f;
			gammaPart.copyTo(ch, mask);
			linearPart.copyTo(ch, ~mask);
		}
	}

	cv::Mat result;
	cv::merge(channels, result); // 合并通道
	return result;
}

LightFieldPtr LFLoad::loadSAI(const std::string &path, bool isRGB) {
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("loadSAI: file not exist! Path: " + path);
	}

	auto start = std::chrono::high_resolution_clock::now();

	// 获取所有文件名
	std::vector<std::string> filenames;
	for (const auto &entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(entry.path().filename().string());
		}
	}
	std::sort(filenames.begin(), filenames.end());

	std::vector<cv::Mat> temp(filenames.size());
	std::vector<std::future<cv::Mat>> futures;

	for (const auto &i : filenames) {
		std::string filename = path + "/" + i;
		futures.push_back(
			std::async(std::launch::async, [&, isRGB, filename]() {
				return cv::imread(filename,
								  isRGB ? cv::IMREAD_COLOR
										: cv::IMREAD_GRAYSCALE); // 写入对应位置
			}));
	}

	for (int i = 0; i < futures.size(); ++i) {
		temp[i] = futures[i].get();
		temp[i].convertTo(temp[i], CV_32FC(temp[i].channels()), 1.0 / 255.0);
	}

	std::cout << "Loading finished!" << std::endl;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

	return std::make_shared<LightFieldData>(std::move(temp));
}
int LFLoad::loadRaw(const std::string &path, cv::Mat &dst, int width,
					int height, int bitDepth) {
	std::vector<uint8_t> raw = readRawFile(path);
	cv::Mat mat;
	if (bitDepth == 8) {
		dst = convertRaw8ToMat(raw.data(), width, height, false);
	} else if (bitDepth == 10) {
		dst = convertRaw10ToMat(raw.data(), width, height, false);
	} else {
		std::cerr << "Unsupported bit depth: " << bitDepth << std::endl;
		return -1;
	}
	cv::cvtColor(mat, dst, LYTRO_BAYER);

	return 0;
}
std::string LFLoad::readSection(std::ifstream &file) {
	std::string section;
	constexpr unsigned char LFM_HEADER[] = {0x89, 'L',	'F',  'M',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00};
	constexpr unsigned char LFC_HEADER[] = {0x89, 'L',	'F',  'C',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00};
	constexpr size_t HEADER_LEN = 12;
	constexpr size_t SHA1_LEN = 45;
	constexpr size_t SHA_PADDING_LEN = 35;
	char header_buf[HEADER_LEN];
	file.read(header_buf, HEADER_LEN);
	if (file.gcount() != HEADER_LEN)
		throw std::runtime_error("readSection: Failed to read section header");

	if (std::memcmp(header_buf, LFM_HEADER, HEADER_LEN) != 0
		&& std::memcmp(header_buf, LFC_HEADER, HEADER_LEN) != 0)
		throw std::runtime_error("readSection: Invalid section header format");

	uint8_t len_bytes[4];
	file.read(reinterpret_cast<char *>(len_bytes), 4);
	if (file.gcount() != 4)
		throw std::runtime_error("readSection: Failed to read section length");

	int section_length = (len_bytes[0] << 24) | (len_bytes[1] << 16)
						 | (len_bytes[2] << 8) | len_bytes[3];

	std::string sha1(SHA1_LEN, '\0');
	file.read(&sha1[0], SHA1_LEN);
	if (file.gcount() != SHA1_LEN)
		throw std::runtime_error("readSection: Failed to read SHA1 hash");

	std::string padding(SHA_PADDING_LEN, '\0');
	file.read(&padding[0], SHA_PADDING_LEN);
	if (file.gcount() != SHA_PADDING_LEN)
		throw std::runtime_error("readSection: Failed to read SHA1 padding");
	if (!std::all_of(padding.begin(), padding.end(),
					 [](const char c) { return c == 0; }))
		throw std::runtime_error(
			"readSection: Non-zero padding found after SHA1");

	section.resize(section_length);
	file.read(&section[0], section_length);
	if (file.gcount() != section_length)
		throw std::runtime_error("readSection: Failed to read section payload");

	unsigned char hash[SHA_DIGEST_LENGTH];
	SHA1(reinterpret_cast<const unsigned char *>(section.data()),
		 section.size(), hash);

	std::ostringstream computed;
	for (unsigned char i : hash) {
		computed << std::hex << std::setw(2) << std::setfill('0')
				 << static_cast<int>(i);
	}
	std::string computed_hash = computed.str();
	std::string given_hash = sha1.substr(5);

	if (computed_hash != given_hash)
		throw std::runtime_error("readSection: SHA1 mismatch");

	// Skip any padding
	while (true) {
		char b;
		file.get(b);
		if (file.eof() || b != '\x00') {
			if (!file.eof())
				file.seekg(-1, std::ios_base::cur);
			break;
		}
	}

	return section;
}
std::vector<std::string> LFLoad::readLytroFile(const std::string &path) {
	std::vector<std::string> sections;
	constexpr unsigned char LFP_HEADER[] = {0x89, 'L',	'F',  'P',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x01};
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("readLytroFile: Cannot open file: " + path);

	char file_header_buf[12];
	file.read(file_header_buf, 12);
	if (file.gcount() != 12
		|| std::memcmp(file_header_buf, LFP_HEADER, 12) != 0)
		throw std::runtime_error("readLytroFile: Invalid file header: " + path);

	uint8_t len_bytes[4];
	file.read(reinterpret_cast<char *>(len_bytes), 4);
	if (file.gcount() != 4)
		throw std::runtime_error(
			"readLytroFile: Failed to read LFP header length");

	int header_len = (len_bytes[0] << 24) | (len_bytes[1] << 16)
					 | (len_bytes[2] << 8) | len_bytes[3];
	if (header_len != 0)
		throw std::runtime_error("readLytroFile: Unexpected LFP header length: "
								 + std::to_string(header_len));

	while (!file.eof()) {
		std::streampos pos = file.tellg();
		char b;
		file.get(b);
		if (file.eof())
			break;
		file.seekg(pos); // rewind

		try {
			std::string section = readSection(file);
			sections.push_back(std::move(section));
		} catch (const std::exception &e) {
			throw std::runtime_error(
				std::string("readLytroFile: Failed to read section: ")
				+ e.what());
		}
	}

	file.close();
	return sections;
}
json LFLoad::readJson(const std::vector<std::string> &sections) {
	json json_dict;
	for (const auto &section : sections) {
		try {
			json parsed = json::parse(section);
			json_dict.update(parsed);
		} catch (...) {
			continue; // 忽略无法解析的部分
		}
	}
	return json_dict;
}
bool LFLoad::hasExtension(const std::string &filename, const std::string &ext) {
	size_t pos = filename.rfind('.');
	if (pos == std::string::npos)
		return false;
	std::string file_ext = filename.substr(pos + 1);
	return file_ext == ext;
}
cv::Mat LFLoad::loadImageFile(const std::string &filename, int width,
							  int height) {
	std::filesystem::path path(filename);
	std::string ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp"
		|| ext == ".tif" || ext == ".tiff") {
		cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
		return gammaConvert(image, true);
	} else if (ext == ".raw") {
		return convertRaw10ToMat(readRawFile(filename).data(), width, height,
								 false);
	} else if (ext == ".lfp" || ext == ".lfr") {
		auto sections = readLytroFile(filename);
		auto buffer = sections[LYTRO_BUFFER_INDEX];
		// json js = LFLoad::readJson(sections);
		cv::Mat lytro_bayer_image = LFLoad::convertRaw10ToMat(
			reinterpret_cast<uint8_t *>(buffer.data()), 7728, 5368, false);
		return lytro_bayer_image;
	} else {
		throw std::runtime_error("loadImageFile: Unsupported file extension: "
								 + ext);
	}
}
