#ifndef RAWDECODE_H
#define RAWDECODE_H

#include "config.h"
#include "json.hpp"

#include <opencv2/core.hpp>
#include <string>

using json = nlohmann::json;

struct Lytro {
	static constexpr int width = 7728;
	static constexpr int height = 5368;
	static constexpr int buffer_index = 10;
};

class RawDecoder {
public:
	explicit RawDecoder();
	cv::Mat decode(std::string filename);
	cv::Mat decode_lytro(std::string filename);
	cv::Mat decode_raw(std::string filename);

	std::vector<std::string> read_lytro_file(const std::string &filename);
	std::string read_section(std::ifstream &file);
	json filter_lfp_json(const json &jsonDict);
	json extract_json(const std::vector<std::string> &sections);

	std::vector<uint8_t> read_raw_file(const std::string &filename);

	cv::Mat unpack_raw2bayer(const uint8_t *src, int width = Lytro::width,
							 int height = Lytro::height);

	std::vector<uint8_t> buffer;
	int width, height;
	json json_dict, lfp;
};

#endif