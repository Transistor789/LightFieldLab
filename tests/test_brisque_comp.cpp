#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief 计算指定目录下所有子孔径图像的平均 BRISQUE 分数
 * @param folder 图像目录
 * @param is_lftoolbox 是否为 LFToolbox 导出的 15x15 图像
 * @param modelPath 模型文件路径
 * @param rangePath 范围文件路径
 */
double calculateAverageBrisque(const std::string &folder, bool is_lftoolbox, const std::string &modelPath,
							   const std::string &rangePath) {
	auto brisque = cv::quality::QualityBRISQUE::create(modelPath, rangePath);
	std::vector<double> scores;

	if (is_lftoolbox) {
		// LFToolbox 15x15 -> 提取中心 9x9 (起始索引为 3, 结束索引为 11)
		for (int r = 3; r < 12; ++r) {
			for (int c = 3; c < 12; ++c) {
				int camIdx = r * 15 + c + 1; // 1-based index
				std::string filename = std::format("input_Cam{:03d}.bmp", camIdx);
				fs::path filePath = fs::path(folder) / filename;

				cv::Mat img = cv::imread(filePath.string());

				if (!img.empty()) {
					if (img.cols % 2 != 0 || img.rows % 2 != 0) {
						cv::resize(img, img, cv::Size(img.cols & ~1, img.rows & ~1), 0, 0, cv::INTER_AREA);
					}

					// 2. 检查极值，排查是否过度截断
					double minVal, maxVal;
					cv::minMaxLoc(img, &minVal, &maxVal);

					cv::Scalar score = brisque->compute(img);
					if (score[0] == 0.0) {
						// std::cerr << std::format("[Warning] Score 0.0 at: {}, Min: {:.1f}, Max: {:.1f}", filename,
						// 						 minVal, maxVal)
						// 		  << std::endl;
					} else {
						scores.push_back(score[0]);
					}
				}
			}
		}
	} else {
		// 自己的软件 9x9 直接遍历
		for (int i = 1; i <= 81; ++i) {
			std::string filename = std::format("input_Cam{:03d}.bmp", i);
			fs::path filePath = fs::path(folder) / filename;

			cv::Mat img = cv::imread(filePath.string());
			if (!img.empty()) {
				cv::Scalar score = brisque->compute(img);
				scores.push_back(score[0]);
			}
		}
	}

	if (scores.empty())
		return -1.0;
	return std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
}

int main(int argc, char **argv) {
	std::string model = "../data/brisque_model_live.yml";
	std::string range = "../data/brisque_range_live.yml";

	std::vector<std::string> lftoolboxFolder = {
		"D:\\code\\light-field-TB\\Results_saving\\toy\\LFColorCorrectedImage",
		// "D:\\code\\light-field-TB\\Results_saving\\rawthreelevel\\LFColorCorrectedImage",
		// "D:\\code\\light-field-TB\\Results_saving\\books\\LFColorCorrectedImage",
	};
	std::vector<std::string> mySoftwareFolder = {
		"D:\\code\\LightFieldLab\\build\\toy",
		// "D:\\code\\LightFieldLab\\build\\rawthreelevel",
		// "D:\\code\\LightFieldLab\\build\\books",
	};

	std::cout << "--- Light Field Image Quality Assessment (BRISQUE) ---" << std::endl;
	std::cout << "Processing folders..." << std::endl;

	for (int i = 0; i < lftoolboxFolder.size(); ++i) {
		double scoreLF = calculateAverageBrisque(lftoolboxFolder[i], true, model, range);
		double scoreMine = calculateAverageBrisque(mySoftwareFolder[i], false, model, range);

		std::cout << std::format("\n[LFToolbox (Center 9x9)] Avg Score: {:.4f}", scoreLF) << std::endl;
		std::cout << std::format("[My Software (Direct 9x9)] Avg Score: {:.4f}", scoreMine) << std::endl;

		// BRISQUE 分数越低，代表图像质量越自然/越好
		std::cout << "\nNote: Lower BRISQUE score indicates better natural "
					 "image quality."
				  << std::endl;
	}

	return 0;
}