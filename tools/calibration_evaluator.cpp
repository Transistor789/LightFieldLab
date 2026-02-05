#include "centers_extract.h"
#include "json.hpp"
#include "lfcalibrate.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

// 存储单张图像结果的结构体
struct EvalResult {
	std::string fileName;
	std::string arrangement;
	std::string resolution;
	double truePitch;
	double mae;
};

// 加载 JSON 真值数据
struct GroundTruth {
	double pitch;
	std::string arrangement;
	std::vector<cv::Point2f> centers;
};

GroundTruth loadGT(const std::string &path) {
	std::ifstream f(path);
	if (!f.is_open())
		throw std::runtime_error("Could not open JSON: " + path);
	json j;
	f >> j;

	GroundTruth gt;
	gt.pitch = j["metadata"]["pitch"];
	gt.arrangement = j["metadata"]["arrangement"];
	for (auto &p : j["centers"]) {
		gt.centers.emplace_back(p["x"], p["y"]);
	}
	return gt;
}

// 绝对坐标误差计算函数
double calculateMAE(const std::vector<cv::Point2f> &gtCenters, const std::pair<cv::Mat, cv::Mat> &detectedMaps,
					double pitch) {
	cv::Mat mapX = detectedMaps.first;
	cv::Mat mapY = detectedMaps.second;
	std::vector<double> errors;
	double threshold = pitch / 2.0;

	for (const auto &gtPt : gtCenters) {
		double minSqDist = std::numeric_limits<double>::max();
		// 寻找最近点
		for (int r = 0; r < mapX.rows; ++r) {
			const float *ptrX = mapX.ptr<float>(r);
			const float *ptrY = mapY.ptr<float>(r);
			for (int c = 0; c < mapX.cols; ++c) {
				float dx = ptrX[c] - gtPt.x;
				float dy = ptrY[c] - gtPt.y;
				double d2 = dx * dx + dy * dy;
				if (d2 < minSqDist)
					minSqDist = d2;
			}
		}
		double dist = std::sqrt(minSqDist);
		if (dist < threshold)
			errors.push_back(dist);
	}
	if (errors.empty())
		return -1.0;
	return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

int main() {
	std::string saveDir = "LightFieldLab_Detected_Results";
	fs::create_directories(saveDir);

	std::string datasetPath = "F:/Light Field Dataset/Sync/synthetic_dataset_100";
	std::string outputCsv = "LightFieldLab.csv";
	std::vector<EvalResult> allResults;

	std::cout << "Starting Batch Evaluation..." << std::endl;

	// 1. 遍历文件夹寻找 PNG 图像
	for (const auto &entry : fs::directory_iterator(datasetPath)) {
		std::string path = entry.path().string();
		if (path.find(".png") == std::string::npos || path.find("gt") != std::string::npos)
			continue;

		std::string imgName = entry.path().filename().string();
		std::string baseName = imgName.substr(0, imgName.find_last_of("."));
		std::string jsonPath = datasetPath + "/" + baseName + "_gt.json";

		try {
			// 2. 加载真值与图像
			GroundTruth gt = loadGT(jsonPath);
			cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
			if (img.empty())
				continue;

			// 3. 配置并运行标定
			LFCalibrate calibrator(img);
			CalibrateConfig config;
			config.diameter = gt.pitch;
			config.autoEstimate = false;
			config.ceMethod = ExtractMethod::LOG_NMS; // 与你之前的输出保持一致
			config.orientation = (gt.arrangement == "hex-col") ? Orientation::VERT : Orientation::HORZ;
			config.hexgridfit = true; // 启用拟合以确保精度

			calibrator.run(img, config);

			// 4. 获取结果并计算 MAE
			auto maps = calibrator.getPoints();
			double mae = calculateMAE(gt.centers, maps, gt.pitch);

			// 5. 记录数据
			EvalResult res;
			res.fileName = imgName;
			res.arrangement = gt.arrangement;
			res.resolution = std::to_string(img.cols) + "x" + std::to_string(img.rows);
			res.truePitch = gt.pitch;
			res.mae = mae;
			allResults.push_back(res);

			std::cout << std::format("[Processed] {} | MAE: {:.6f} px", imgName, mae) << std::endl;

			// --- 6. 保存检测到的中心点为 JSON ---
			json outputJson;

			// 重新读取原始 JSON 以获取完整的 metadata
			std::ifstream originalF(jsonPath);
			json originalJson;
			originalF >> originalJson;
			outputJson["metadata"] = originalJson["metadata"];

			// 构造 centers 数组
			json centersArray = json::array();
			int ptId = 0;
			for (int r = 0; r < maps.first.rows; ++r) {
				const float *ptrX = maps.first.ptr<float>(r);
				const float *ptrY = maps.second.ptr<float>(r);
				for (int c = 0; c < maps.first.cols; ++c) {
					centersArray.push_back({{"id", ptId++}, {"x", ptrX[c]}, {"y", ptrY[c]}});
				}
			}
			outputJson["centers"] = centersArray;

			// 写入本地文件
			std::ofstream outF(saveDir + "/" + baseName + ".json");
			outF << outputJson.dump(4); // 缩进 4 格
			outF.close();

		} catch (const std::exception &e) {
			std::cerr << "Error processing " << imgName << ": " << e.what() << std::endl;
		}
	}

	// 6. 写入 CSV 文件
	std::ofstream csv(outputCsv);
	csv << "FileName,Arrangement,Resolution,Pitch,Error(MAE)\n";

	double totalMAE = 0;
	int validCount = 0;

	for (const auto &r : allResults) {
		csv << r.fileName << "," << r.arrangement << "," << r.resolution << "," << std::fixed << std::setprecision(6)
			<< r.truePitch << "," << r.mae << "\n";

		if (r.mae >= 0) {
			totalMAE += r.mae;
			validCount++;
		}
	}

	// 7. 写入均值行
	if (validCount > 0) {
		double avgMAE = totalMAE / validCount;
		csv << "\nTOTAL_AVERAGE,,,," << avgMAE << "\n";
		std::cout << "\n========================================" << std::endl;
		std::cout << "Evaluation Complete. Average MAE: " << avgMAE << " px" << std::endl;
		std::cout << "Results saved to: " << outputCsv << std::endl;
	}

	return 0;
}