#include "raw_decode.h"

#include "config.h"
#include "utils.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <openssl/sha.h>
#include <string>
#include <vector>

RawDecoder::RawDecoder() {}

cv::Mat RawDecoder::decode(std::string filename) {
	cv::Mat ret;
	std::filesystem::path path(filename);
	std::string ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	if (ext == ".lfp" || ext == ".lfr") {
		ret = decode_lytro(filename);
	} else if (ext == ".raw") {
		ret = decode_raw(filename);
	}
	return ret;
}

cv::Mat RawDecoder::decode_lytro(std::string filename) {
	// decode lfp file
	auto sections = read_lytro_file(filename);
	// retrieve JSON data
	json_dict = extract_json(sections);
	// JSON file export
	writeJson(get_base_filename(filename) + ".json", json_dict);
	// writeJson("../data/" + get_base_filename(filename) + ".json", json_dict);
	// decompose JSON data
	width = json_dict["image"]["width"];
	height = json_dict["image"]["height"];
	// filter LFP metadata settings
	lfp = filter_lfp_json(json_dict);

	// compose bayer image from lfp file
	auto img_buf =
		reinterpret_cast<uint8_t *>(sections[Lytro::buffer_index].data());
	return unpack_raw2bayer(img_buf, width, height);
}

cv::Mat RawDecoder::decode_raw(std::string filename) {
	buffer = read_raw_file(filename);
	auto ptr = buffer.data();
	return unpack_raw2bayer(ptr);
}

cv::Mat RawDecoder::unpack_raw2bayer(const uint8_t *src, int width,
									 int height) {
	if (src == nullptr) {
		throw std::runtime_error("convertRaw10ToMat: src is nullptr!");
	}
	cv::Mat ret = cv::Mat(height, width, CV_16UC1);
	auto *ptr = ret.ptr<uint16_t>();
	int dataSize = width * height * 5 / 4;
	for (int i = 0, index = 0; i < dataSize; i += 5, index += 4) {
		const uint8_t low_bits = src[i + 4];
		ptr[index] = (src[i] << 2) | ((low_bits >> 6) & 0x03);
		ptr[index + 1] = (src[i + 1] << 2) | ((low_bits >> 4) & 0x03);
		ptr[index + 2] = (src[i + 2] << 2) | ((low_bits >> 2) & 0x03);
		ptr[index + 3] = (src[i + 3] << 2) | (low_bits & 0x03);
	}
	return ret;
}

std::vector<std::string> RawDecoder::read_lytro_file(
	const std::string &filename) {
	std::vector<std::string> sections;
	constexpr unsigned char LFP_HEADER[] = {0x89, 'L',	'F',  'P',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x01};
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("readLytroFile: Cannot open file: "
								 + filename);

	char file_header_buf[12];
	file.read(file_header_buf, 12);
	if (file.gcount() != 12
		|| std::memcmp(file_header_buf, LFP_HEADER, 12) != 0)
		throw std::runtime_error("readLytroFile: Invalid file header: "
								 + filename);

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
			std::string section = read_section(file);
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

std::vector<uint8_t> RawDecoder::read_raw_file(const std::string &filename) {
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

json RawDecoder::extract_json(const std::vector<std::string> &sections) {
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

std::string RawDecoder::read_section(std::ifstream &file) {
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

json RawDecoder::filter_lfp_json(const json &jsonDict) {
	json settings;

	const std::vector<std::string> channels = {"b", "r", "gb", "gr"};

	// 获取 serialNumber 或 model
	std::string serial;
	if (jsonDict.contains("camera")
		&& jsonDict["camera"].contains("serialNumber")) {
		serial = jsonDict["camera"]["serialNumber"];
	}

	std::string camModel = serial.empty()
							   ? jsonDict["camera"]["model"].get<std::string>()
							   : serial;

	if (camModel.empty()) {
		throw std::runtime_error("Camera model is empty");
	}

	// 第一代 Lytro: Axxx or Fxxx
	if (camModel[0] == 'A' || camModel[0] == 'F') {
		int bit =
			jsonDict["image"]["rawDetails"]["pixelPacking"]["bitsPerPixel"];
		if (bit != 12) {
			throw std::runtime_error(
				"Unrecognized bit packing format (expected 12-bit for Gen1)");
		}

		settings["bit"] = bit;
		settings["bay"] = std::string("BGGR"); // Gen1 相机通常是 BGGR 模式

		// --- 1. 黑电平/白电平 (BLC) ---
		// Gen1 的 JSON 结构通常不提供 BLC/White point，使用硬编码的 12-bit
		// 默认值 12-bit 最大值: 2^12 - 1 = 4095
		uint16_t white_12bit = 4095;
		uint16_t black_12bit = 0;

		// 严格遵循 Gen2 的结构，使用 vector<uint16_t>
		std::vector<uint16_t> black_vec = {black_12bit, black_12bit,
										   black_12bit, black_12bit};
		std::vector<uint16_t> white_vec = {white_12bit, white_12bit,
										   white_12bit, white_12bit};

		settings["blc"]["black"] = black_vec;
		settings["blc"]["white"] = white_vec;

		// --- 2. AWB 增益 (转换为 Gen2 的 [Gr, R, B, Gb] 顺序和类型) ---

		// Gen1 JSON 存储为 ["b", "r", "gb", "gr"] 的结构
		auto &wb_json = jsonDict["image"]["color"]["whiteBalanceGain"];
		std::vector<float> awb_gains = {// Index 0: B
										wb_json.value("b", 1.0f),
										// Index 1: Gr
										wb_json.value("gr", 1.0f),
										// Index 2: Gb
										wb_json.value("gb", 1.0f),
										// Index 3: R
										wb_json.value("r", 1.0f)};
		settings["awb"] = awb_gains;

		// --- 3. CCM & Gamma ---
		// CCM: 确保数据类型为 std::vector<float>
		settings["ccm"] = jsonDict["image"]["color"]["ccmRgbToSrgbArray"]
							  .get<std::vector<float>>();

		// Gamma: 确保数据类型为 float
		settings["gam"] = jsonDict["image"]["color"]["gamma"].get<float>();

		// Gen1 JSON 通常没有
		// modulationExposureBias，但为了结构一致性，可以设为默认值
		settings["exp"] = 0.0f;

	}
	// 第二代 Lytro: Bxxx, Ixxx, or all-digit
	else if (camModel[0] == 'B' || camModel[0] == 'I'
			 || std::all_of(camModel.begin(), camModel.end(), ::isdigit)) {
		int bit = jsonDict["image"]["pixelPacking"]["bitsPerPixel"];
		if (bit != 10) {
			throw std::runtime_error(
				"Unrecognized bit packing format (expected 10-bit for "
				"Gen2)");
		}
		settings["bit"] = bit;
		settings["bay"] = std::string("GRBG");

		auto &fmt = jsonDict["image"]["pixelFormat"];
		std::vector<uint16_t> black_vec = {
			fmt["black"].value("gr", (uint16_t)0),
			fmt["black"].value("r", (uint16_t)0),
			fmt["black"].value("b", (uint16_t)0),
			fmt["black"].value("gb", (uint16_t)0)};
		std::vector<uint16_t> white_vec = {
			fmt["white"].value("gr", (uint16_t)1023),
			fmt["white"].value("r", (uint16_t)1023),
			fmt["white"].value("b", (uint16_t)1023),
			fmt["white"].value("gb", (uint16_t)1023)};
		settings["blc"]["black"] = black_vec;
		settings["blc"]["white"] = white_vec;

		auto &wb_json = jsonDict["image"]["color"]["whiteBalanceGain"];
		std::vector<float> awb_gains = {
			wb_json.value("gr", 1.0f), // Index 0: Gr (Green in Red row)
			wb_json.value("r", 1.0f),  // Index 1: R  (Red)
			wb_json.value("b", 1.0f),  // Index 2: B  (Blue)
			wb_json.value("gb", 1.0f)  // Index 3: Gb (Green in Blue row)
		};
		settings["awb"] = awb_gains;
		settings["ccm"] =
			jsonDict["image"]["color"]["ccm"].get<std::vector<float>>();
		settings["gam"] =
			jsonDict["master"]["picture"]["frameArray"][0]["frame"]["metadata"]
					["image"]["color"]["gamma"]
						.get<float>();
		settings["exp"] =
			jsonDict["image"]["modulationExposureBias"].get<float>();

	} else {
		throw std::runtime_error("Unsupported camera model: " + camModel);
	}

	return settings;
}