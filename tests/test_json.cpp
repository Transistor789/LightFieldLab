#include "json.hpp"

#include <fstream>
#include <iostream>

using json = nlohmann::json;

int main() {
	std::ifstream file("../data/test_data.json");
	json j = json::parse(file);
	// std::cout << j.dump() << std::endl;
	std::cout << j["points"].dump(4) << std::endl;
	std::cout << j["points"]["cols"].dump(4) << std::endl;
	std::cout << j["points"]["rows"].dump(4) << std::endl;
	std::cout << j["points"]["data"].dump(4) << std::endl;

	std::ofstream f("../../data/save.json");
	f << j.dump();

	return 0;
}