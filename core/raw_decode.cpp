#include "raw_decode.h"

#include "utils.h" // 包含 get_base_filename

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip> // for std::setw
#include <iostream>
#include <omp.h>
#include <openssl/sha.h>
#include <sstream> // for std::ostringstream

namespace fs = std::filesystem;

// 匿名命名空间存放仅本文件可见的常量
namespace {
constexpr int LYTRO_WIDTH = 7728;
constexpr int LYTRO_HEIGHT = 5368;
constexpr int BUFFER_INDEX = 10;
} // namespace

// =============================================================
// 公开接口实现
// =============================================================

cv::Mat RawDecoder::DecodeLytro(const std::string &filename,
								json &outMetadata) {
	// 1. 读取文件段
	auto sections = ReadLytroFile(filename);

	// 2. 提取 JSON
	json fullDict = ExtractJson(sections);

	// 可选：导出 JSON
	// writeJson(get_base_filename(filename) + ".json", fullDict);

	// 3. 过滤关键设置并通过引用传出
	outMetadata = FilterLfpJson(fullDict);

	// 获取尺寸
	int w = fullDict["image"].value("width", LYTRO_WIDTH);
	int h = fullDict["image"].value("height", LYTRO_HEIGHT);

	// 4. 解包图像
	if (sections.size() <= BUFFER_INDEX) {
		throw std::runtime_error(
			"LFP file structure invalid: missing image data");
	}

	// 直接操作内存，无额外拷贝
	const uint8_t *imgBuf =
		reinterpret_cast<const uint8_t *>(sections[BUFFER_INDEX].data());
	return UnpackRaw10ToBayer(imgBuf, w, h);
}

cv::Mat RawDecoder::DecodeRaw(const std::string &filename) {
	// 读取文件到临时 vector
	std::vector<uint8_t> buffer = ReadRawFile(filename);
	return UnpackRaw10ToBayer(buffer.data(), LYTRO_WIDTH, LYTRO_HEIGHT);
}

cv::Mat RawDecoder::DecodeWhiteImage(const std::string &filename,
									 json &outMetadata) {
	// 1. 复用 DecodeRaw 读取像素
	cv::Mat raw = DecodeRaw(filename);
	if (raw.empty())
		return raw;

	// 2. 加载元数据
	outMetadata = LoadWhiteMetadata(filename);
	if (outMetadata.is_null() || outMetadata.empty()) {
		std::cerr << "[RawDecoder] Warning: No metadata for white image: "
				  << filename << std::endl;
		return raw;
	}

	// 3. 应用白平衡和黑电平
	// ApplyWhiteBalance(raw, outMetadata);

	return raw;
}

// =============================================================
// 私有辅助函数实现
// =============================================================

std::vector<std::string> RawDecoder::ReadLytroFile(
	const std::string &filename) {
	std::vector<std::string> sections;
	constexpr unsigned char LFP_HEADER[] = {0x89, 'L',	'F',  'P',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x01};

	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("ReadLytroFile: Cannot open file: "
								 + filename);

	char headerBuf[12];
	file.read(headerBuf, 12);
	if (file.gcount() != 12 || std::memcmp(headerBuf, LFP_HEADER, 12) != 0) {
		throw std::runtime_error("ReadLytroFile: Invalid LFP header: "
								 + filename);
	}

	// 跳过 header length (4 bytes)
	uint8_t lenBytes[4];
	file.read(reinterpret_cast<char *>(lenBytes), 4);
	if (file.gcount() != 4)
		throw std::runtime_error(
			"ReadLytroFile: Failed to read LFP header length");

	int headerLen = (lenBytes[0] << 24) | (lenBytes[1] << 16)
					| (lenBytes[2] << 8) | lenBytes[3];
	if (headerLen != 0)
		throw std::runtime_error("ReadLytroFile: Unexpected LFP header length: "
								 + std::to_string(headerLen));

	while (!file.eof()) {
		std::streampos pos = file.tellg();
		char b;
		file.get(b); // 探测 EOF
		if (file.eof())
			break;
		file.seekg(pos);

		try {
			sections.emplace_back(ReadSection(file));
		} catch (const std::exception &e) {
			throw std::runtime_error(
				std::string("ReadLytroFile: Failed to read section: ")
				+ e.what());
		}
	}
	return sections;
}

std::string RawDecoder::ReadSection(std::ifstream &file) {
	constexpr unsigned char LFM_HEADER[] = {0x89, 'L',	'F',  'M',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00};
	constexpr unsigned char LFC_HEADER[] = {0x89, 'L',	'F',  'C',	0x0d, 0x0a,
											0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00};
	constexpr size_t HEADER_LEN = 12;
	constexpr size_t SHA1_LEN = 45;
	constexpr size_t PAD_LEN = 35;

	char headerBuf[HEADER_LEN];
	file.read(headerBuf, HEADER_LEN);
	if (file.gcount() != HEADER_LEN)
		throw std::runtime_error("ReadSection: Failed to read section header");

	if (std::memcmp(headerBuf, LFM_HEADER, HEADER_LEN) != 0
		&& std::memcmp(headerBuf, LFC_HEADER, HEADER_LEN) != 0)
		throw std::runtime_error("ReadSection: Invalid section header format");

	uint8_t lenBytes[4];
	file.read(reinterpret_cast<char *>(lenBytes), 4);
	if (file.gcount() != 4)
		throw std::runtime_error("ReadSection: Failed to read section length");

	int sectionLen = (lenBytes[0] << 24) | (lenBytes[1] << 16)
					 | (lenBytes[2] << 8) | lenBytes[3];

	std::string sha1(SHA1_LEN, '\0');
	file.read(&sha1[0], SHA1_LEN);
	if (file.gcount() != SHA1_LEN)
		throw std::runtime_error("ReadSection: Failed to read SHA1 hash");

	std::string padding(PAD_LEN, '\0');
	file.read(&padding[0], PAD_LEN);
	if (file.gcount() != PAD_LEN)
		throw std::runtime_error("ReadSection: Failed to read SHA1 padding");

	if (!std::all_of(padding.begin(), padding.end(),
					 [](const char c) { return c == 0; }))
		throw std::runtime_error(
			"ReadSection: Non-zero padding found after SHA1");

	std::string section(sectionLen, '\0');
	file.read(&section[0], sectionLen);
	if (file.gcount() != sectionLen)
		throw std::runtime_error("ReadSection: Failed to read section payload");

	// 校验 SHA1
	unsigned char hash[SHA_DIGEST_LENGTH];
	SHA1(reinterpret_cast<const unsigned char *>(section.data()),
		 section.size(), hash);

	std::ostringstream computed;
	for (unsigned char i : hash) {
		computed << std::hex << std::setw(2) << std::setfill('0')
				 << static_cast<int>(i);
	}
	std::string computedHash = computed.str();
	// LFP 文件中的 SHA1 字符串通常以 "sha1-" 开头，截取后40位比较
	// 如果文件中存储的是纯哈希字符串，则直接比较。根据原代码逻辑 sha1.substr(5)
	std::string givenHash = sha1.substr(5);

	if (computedHash != givenHash) {
		// 有些实现可能会忽略校验错误，但为了严谨这里抛出异常
		throw std::runtime_error("ReadSection: SHA1 mismatch");
	}

	// Skip trailing padding (null bytes)
	while (true) {
		char b;
		file.get(b);
		if (file.eof() || b != '\x00') {
			if (!file.eof())
				file.seekg(-1, std::ios::cur);
			break;
		}
	}
	return section;
}

std::vector<uint8_t> RawDecoder::ReadRawFile(const std::string &filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file)
		throw std::runtime_error("File not found: " + filename);

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer(size);
	if (!file.read((char *)buffer.data(), size))
		throw std::runtime_error("Read error");
	return buffer;
}

json RawDecoder::ExtractJson(const std::vector<std::string> &sections) {
	json combined;
	for (const auto &sec : sections) {
		try {
			// 尝试解析每一段，忽略非 JSON 段
			json j = json::parse(sec, nullptr, false);
			if (!j.is_discarded()) {
				combined.update(j);
			}
		} catch (...) {
		}
	}
	return combined;
}

json RawDecoder::FilterLfpJson(const json &jsonDict) {
	json settings;

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
		uint16_t white12bit = 4095;
		uint16_t black12bit = 0;

		std::vector<uint16_t> blackVec = {black12bit, black12bit, black12bit,
										  black12bit};
		std::vector<uint16_t> whiteVec = {white12bit, white12bit, white12bit,
										  white12bit};

		settings["blc"]["black"] = blackVec;
		settings["blc"]["white"] = whiteVec;

		// --- 2. AWB 增益 (转换为 Gen2 的 [Gr, R, B, Gb] 顺序和类型) ---
		auto &wbJson = jsonDict["image"]["color"]["whiteBalanceGain"];
		std::vector<float> awbGains = {
			wbJson.value("b", 1.0f),  // Index 0: B
			wbJson.value("gr", 1.0f), // Index 1: Gr
			wbJson.value("gb", 1.0f), // Index 2: Gb
			wbJson.value("r", 1.0f)	  // Index 3: R
		};
		settings["awb"] = awbGains;

		// --- 3. CCM & Gamma ---
		settings["ccm"] = jsonDict["image"]["color"]["ccmRgbToSrgbArray"]
							  .get<std::vector<float>>();
		settings["gam"] = jsonDict["image"]["color"]["gamma"].get<float>();
		settings["exp"] = 0.0f;

	}
	// 第二代 Lytro: Bxxx, Ixxx, or all-digit
	else if (camModel[0] == 'B' || camModel[0] == 'I'
			 || std::all_of(camModel.begin(), camModel.end(), ::isdigit)) {
		int bit = jsonDict["image"]["pixelPacking"]["bitsPerPixel"];
		if (bit != 10) {
			throw std::runtime_error(
				"Unrecognized bit packing format (expected 10-bit for Gen2)");
		}
		settings["bit"] = bit;
		settings["bay"] = std::string("GRBG");

		// === 提取标定所需的关键元数据 (Serial & GeoRef) ===
		settings["serial"] = serial; // 序列号

		// 提取 Geometry Reference (白图匹配的关键 Hash)
		std::string geoRef;
		try {
			if (camModel[0] == 'B' || camModel[0] == 'I'
				|| std::isdigit(camModel[0])) {
				// Gen2 (Illum): 位于 frames[0].frame.geometryCorrectionRef
				if (jsonDict.contains("frames")
					&& !jsonDict["frames"].empty()) {
					auto &f0 = jsonDict["frames"][0];
					if (f0.contains("frame")
						&& f0["frame"].contains("geometryCorrectionRef")) {
						geoRef = f0["frame"]["geometryCorrectionRef"]
									 .get<std::string>();
					}
				}
			} else {
				// 兼容逻辑 (虽在 Gen2 分支，但保留完整性): 位于
				// picture.derivationArray[0]
				if (jsonDict.contains("picture")
					&& jsonDict["picture"].contains("derivationArray")) {
					auto &arr = jsonDict["picture"]["derivationArray"];
					if (!arr.empty()) {
						geoRef = arr[0].get<std::string>();
					}
				}
			}
		} catch (...) {
			// 忽略异常，保持 geoRef 为空
		}
		settings["geo_ref"] = geoRef;

		// === 提取 ISP 参数 ===
		auto &fmt = jsonDict["image"]["pixelFormat"];
		std::vector<uint16_t> blackVec = {
			fmt["black"].value("gr", (uint16_t)0),
			fmt["black"].value("r", (uint16_t)0),
			fmt["black"].value("b", (uint16_t)0),
			fmt["black"].value("gb", (uint16_t)0)};
		std::vector<uint16_t> whiteVec = {
			fmt["white"].value("gr", (uint16_t)1023),
			fmt["white"].value("r", (uint16_t)1023),
			fmt["white"].value("b", (uint16_t)1023),
			fmt["white"].value("gb", (uint16_t)1023)};
		settings["blc"]["black"] = blackVec;
		settings["blc"]["white"] = whiteVec;

		auto &wbJson = jsonDict["image"]["color"]["whiteBalanceGain"];
		std::vector<float> awbGains = {
			wbJson.value("gr", 1.0f), // Index 0: Gr
			wbJson.value("r", 1.0f),  // Index 1: R
			wbJson.value("b", 1.0f),  // Index 2: B
			wbJson.value("gb", 1.0f)  // Index 3: Gb
		};
		settings["awb"] = awbGains;
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

json RawDecoder::LoadWhiteMetadata(const std::string &rawFilename) {
	fs::path rawPath(rawFilename);
	const std::vector<std::string> exts = {".TXT", ".txt", ".json"};

	for (const auto &ext : exts) {
		fs::path metaPath = rawPath;
		metaPath.replace_extension(ext);
		if (fs::exists(metaPath)) {
			try {
				std::ifstream f(metaPath);
				json j;
				f >> j;
				return j;
			} catch (...) {
			}
		}
	}
	return json(); // 返回空对象
}

cv::Mat RawDecoder::UnpackRaw10ToBayer(const uint8_t *src, int width,
									   int height) {
	if (!src)
		throw std::runtime_error("Source buffer is null");

	cv::Mat ret(height, width, CV_16UC1);

	if (ret.isContinuous()) {
		uint16_t *dst = ret.ptr<uint16_t>(0);
		int totalPixels = width * height;
		int chunks = totalPixels / 4;

#pragma omp parallel for
		for (int i = 0; i < chunks; ++i) {
			int srcIdx = i * 5;
			int dstIdx = i * 4;

			const uint8_t b0 = src[srcIdx];
			const uint8_t b1 = src[srcIdx + 1];
			const uint8_t b2 = src[srcIdx + 2];
			const uint8_t b3 = src[srcIdx + 3];
			const uint8_t low = src[srcIdx + 4];

			dst[dstIdx] = (b0 << 2) | ((low >> 6) & 0x03);
			dst[dstIdx + 1] = (b1 << 2) | ((low >> 4) & 0x03);
			dst[dstIdx + 2] = (b2 << 2) | ((low >> 2) & 0x03);
			dst[dstIdx + 3] = (b3 << 2) | (low & 0x03);
		}
	}
	return ret;
}

void RawDecoder::ApplyWhiteBalance(cv::Mat &raw, const json &j) {
	// 提取逻辑 (复用 FilterLfpJson 的思路，但针对白图结构优化)
	int blcR = 0, blcGr = 0, blcGb = 0, blcB = 0;
	float gainR = 1.0f, gainGr = 1.0f, gainGb = 1.0f, gainB = 1.0f;

	// 简化的辅助 Lambda
	auto getVal = [&](const json &node, const char *key, auto defaultVal) {
		return node.value(key, defaultVal);
	};

	// 提取 Black Level (需要适配 MOD_00xx.TXT 结构)
	// 路径:
	// master.picture.frameArray[0].frame.metadata.image.rawDetails.pixelFormat.black
	try {
		if (j.contains("master")) {
			auto &meta =
				j["master"]["picture"]["frameArray"][0]["frame"]["metadata"];

			// BLC
			if (meta["image"]["rawDetails"]["pixelFormat"].contains("black")) {
				auto &b = meta["image"]["rawDetails"]["pixelFormat"]["black"];
				blcR = getVal(b, "r", 0);
				blcGr = getVal(b, "gr", 0);
				blcGb = getVal(b, "gb", 0);
				blcB = getVal(b, "b", 0);
			}

			// AWB Gains from Normalized Responses
			if (meta["devices"]["sensor"].contains("normalizedResponses")) {
				auto &resp =
					meta["devices"]["sensor"]["normalizedResponses"][0];
				if (resp.contains("r"))
					gainR = 1.0f / resp["r"].get<float>();
				if (resp.contains("gr"))
					gainGr = 1.0f / resp["gr"].get<float>();
				if (resp.contains("b"))
					gainB = 1.0f / resp["b"].get<float>();
				// Gb 通常不存在或设为 1.0
			}
		}
	} catch (...) {
		std::cerr << "[RawDecoder] Metadata parse error, skipping correction."
				  << std::endl;
		return;
	}

	// 应用校正
#pragma omp parallel for
	for (int r = 0; r < raw.rows; r += 2) {
		uint16_t *ptr0 = raw.ptr<uint16_t>(r);
		uint16_t *ptr1 = raw.ptr<uint16_t>(r + 1);
		for (int c = 0; c < raw.cols; c += 2) {
			// GRBG Pattern
			// Row 0: G(gr), R
			ptr0[c] = cv::saturate_cast<uint16_t>(
				std::max(0.0f, (ptr0[c] - blcGr) * gainGr));
			ptr0[c + 1] = cv::saturate_cast<uint16_t>(
				std::max(0.0f, (ptr0[c + 1] - blcR) * gainR));
			// Row 1: B, G(gb)
			ptr1[c] = cv::saturate_cast<uint16_t>(
				std::max(0.0f, (ptr1[c] - blcB) * gainB));
			ptr1[c + 1] = cv::saturate_cast<uint16_t>(
				std::max(0.0f, (ptr1[c + 1] - blcGb) * gainGb));
		}
	}
}