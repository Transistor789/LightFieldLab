#include "centers_extract.h"
#include "json.hpp"
#include "lfcalibrate.h"
#include "lfio.h"

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
	double missingRate;
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
// 修改后的评价函数：同时计算 MAE 和 漏检率
double calculateMAE(const std::vector<cv::Point2f> &gtCenters, const std::pair<cv::Mat, cv::Mat> &detectedMaps,
					double pitch, double &outMissingRate) {
	cv::Mat mapX = detectedMaps.first;
	cv::Mat mapY = detectedMaps.second;
	std::vector<double> errors;
	double threshold = pitch / 2.0;
	int detectedCount = 0;

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
		if (dist < threshold) {
			errors.push_back(dist);
			detectedCount++;
		}
	}

	// 计算漏检率
	outMissingRate = (1.0 - (double)detectedCount / gtCenters.size()) * 100.0;

	if (errors.empty())
		return -1.0;
	return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

void evaluateAllMethods() {
	// 基础路径配置
	const std::string datasetPath = "F:/Light Field Dataset/Results/Calibration/Synthetic/synthetic_dataset_100";

	// 定义待评估的方法名称及其对应的结果文件夹
	const std::vector<std::pair<std::string, std::string>> methods = {
		{"LFToolbox", "F:/Light Field Dataset/Results/Calibration/Synthetic/LFToolbox_Detected_Results"},
		{"Plenopticam", "F:/Light Field Dataset/Results/Calibration/Synthetic/Plenopticam_Detected_Results"},
		{"Ours", "F:/Light Field Dataset/Results/Calibration/Synthetic/LightFieldLab_Detected_Results"}};

	std::cout << "\n==========================================" << std::endl;
	std::cout << "   批量评估开始：标定算法精度对比" << std::endl;
	std::cout << "==========================================" << std::endl;

	for (const auto &[methodName, resultDir] : methods) {
		if (!fs::exists(resultDir)) {
			std::cerr << "[跳过] 目录不存在: " << resultDir << std::endl;
			continue;
		}

		std::cout << "\n[正在处理方法: " << methodName << "]" << std::endl;

		double sumTotalMAE = 0.0;
		double sumTotalMissingRate = 0.0;
		int validFilesCount = 0;

		// 遍历该方法下的所有 JSON 检测结果
		for (const auto &entry : fs::directory_iterator(resultDir)) {
			if (entry.path().extension() != ".json")
				continue;

			std::string baseName = entry.path().stem().string(); // 例如 MLA_001_hex-row
			std::string detectedJsonPath = entry.path().string();
			std::string gtJsonPath = datasetPath + "/" + baseName + "_gt.json";

			if (!fs::exists(gtJsonPath))
				continue;

			try {
				// 1. 加载真值 (Ground Truth)
				GroundTruth gt = loadGT(gtJsonPath);
				double threshold = gt.pitch / 2.0;

				// 2. 加载该工具的检测结果
				std::ifstream f(detectedJsonPath);
				json j;
				f >> j;
				std::vector<cv::Point2f> detPoints;
				for (const auto &p : j["centers"]) {
					detPoints.emplace_back(p["x"].get<float>(), p["y"].get<float>());
				}

				// 3. 将检测点转换为 calculateMAE 能够识别的格式 (pair<Mat, Mat>)
				cv::Mat mapX(1, (int)detPoints.size(), CV_32F);
				cv::Mat mapY(1, (int)detPoints.size(), CV_32F);
				for (size_t k = 0; k < detPoints.size(); ++k) {
					mapX.at<float>(0, (int)k) = detPoints[k].x;
					mapY.at<float>(0, (int)k) = detPoints[k].y;
				}

				// 4. 执行核心误差算法
				double currentMissingRate = 100.0;
				double currentMAE = calculateMAE(gt.centers, {mapX, mapY}, gt.pitch, currentMissingRate);

				if (currentMAE >= 0) {
					sumTotalMAE += currentMAE;
					sumTotalMissingRate += currentMissingRate;
					validFilesCount++;
				}

			} catch (const std::exception &e) {
				// 捕获个别文件损坏的情况，不中断整体评估
			}
		}

		// 输出该方法的平均统计结果
		if (validFilesCount > 0) {
			double finalMAE = sumTotalMAE / validFilesCount;
			double finalMissing = sumTotalMissingRate / validFilesCount;

			std::cout << " >> 成功评估样本数: " << validFilesCount << std::endl;
			std::cout << " >> 平均绝对误差 (MAE): " << std::fixed << std::setprecision(6) << finalMAE << " px"
					  << std::endl;
			std::cout << " >> 平均漏检率 (Missing): " << std::fixed << std::setprecision(3) << finalMissing << " %"
					  << std::endl;
		} else {
			std::cout << " >> 错误：未发现有效可匹配的样本数据。" << std::endl;
		}
	}
	std::cout << "\n==========================================" << std::endl;
	std::cout << "   所有方法评估完成。" << std::endl;
	std::cout << "==========================================" << std::endl;
}
void calibrate() {
	std::string saveDir = "LightFieldLab_Detected_Results";
	fs::create_directories(saveDir);

	std::string datasetPath = "F:/Light Field Dataset/Results/Calibration/Synthetic/synthetic_dataset_100";
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
			config.ceMethod = ExtractMethod::Contour; // 与你之前的输出保持一致
			config.orientation = (gt.arrangement == "hex-col") ? Orientation::VERT : Orientation::HORZ;
			config.bayer = BayerPattern::NONE;		   // 合成数据不需要处理 Bayer 纹理
			config.hexgridfit = false;				   // 启用拟合以确保精度
			std::cout << config.diameter << std::endl; /////
			calibrator.run(img, config);

			// 4. 获取结果并计算 MAE 与 漏检率
			auto maps = calibrator.getPoints();
			double missingRate = 100.0;
			double mae = calculateMAE(gt.centers, maps, gt.pitch, missingRate);

			// 5. 记录数据
			EvalResult res;
			res.fileName = imgName;
			res.arrangement = gt.arrangement;
			res.resolution = std::to_string(img.cols) + "x" + std::to_string(img.rows);
			res.truePitch = gt.pitch;
			res.mae = mae;
			res.missingRate = missingRate; // 记录漏检率
			allResults.push_back(res);

			std::cout << std::format("[Processed] {} | MAE: {:.6f} px | Missing: {:.3f}%", imgName, mae, missingRate)
					  << std::endl;

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
	csv << "FileName,Arrangement,Resolution,Pitch,MAE,MissingRate(%)\n";

	double totalMAE = 0, totalMissing = 0;
	int validCount = 0;

	for (const auto &r : allResults) {
		csv << r.fileName << "," << r.arrangement << "," << r.resolution << "," << std::fixed << std::setprecision(6)
			<< r.truePitch << "," << r.mae << "," << std::setprecision(2) << r.missingRate << "\n";

		if (r.mae >= 0) {
			totalMAE += r.mae;
			totalMissing += r.missingRate;
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
}

void calibrateLytroRealData() {
	// 1. 路径与配置设置
	std::string datasetPath = "F:/Light Field Dataset/LytroIllum_Dataset_INRIA_SIROCCO/B5144000580";
	std::string saveDir = "F:/Light Field Dataset/Results/Calibration/Real/LightFieldLab";
	fs::create_directories(saveDir);

	std::cout << "\n==========================================" << std::endl;
	std::cout << "   开始批量标定 Lytro 真实白图数据" << std::endl;
	std::cout << "==========================================" << std::endl;

	// Lytro Illum 物理先验参数
	// Lytro Illum 的微透镜是列偏移六边形排列 (hex-col，在很多算法里处理为垂直偏置)
	// 且其等效微透镜间距（在原始分辨率 7728x5368 下）约为 14 到 15 个像素。
	const double LYTRO_APPROX_PITCH = 13;
	const std::string arrangement = "hex-row";

	// 2. 遍历指定命名规则的图像 (MOD_0000.RAW 至 MOD_0033.RAW)
	for (int i = 0; i <= 33; ++i) {
		// 构造文件名，例如: MOD_0000.RAW
		std::stringstream ss;
		ss << "MOD_" << std::setfill('0') << std::setw(4) << i << ".RAW";
		std::string fileName = ss.str();
		std::string filePath = datasetPath + "/" + fileName;

		// 检查文件是否存在
		if (!fs::exists(filePath)) {
			std::cerr << "[跳过] 找不到文件: " << fileName << std::endl;
			continue;
		}

		std::string baseName = fileName.substr(0, fileName.find_last_of("."));

		try {
			// 3. 读取 RAW 图像
			// 注意：真实 Lytro 数据可能有 12-bit 或 10-bit 打包问题。
			// 这里假设它已经被解包为普通的 8-bit 或 16-bit 单通道灰度图，或者使用了可以由 imread 读取的格式。
			// 如果你的 .RAW 是纯二进制文件且没有文件头，你需要使用 std::ifstream 手动读取 (7728x5368)。
			// 在此，我们先使用 cv::imread_ANYDEPTH 尝试读取（依赖于你的后续预处理或 Lytro LFP 提取器）。
			// cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
			json j;
			cv::Mat img = LFIO::ReadWhiteImageManual(filePath, j);

			if (img.empty()) {
				std::cerr << "无法通过 cv::imread 解码图像: " << fileName << "，请检查 RAW 文件读取方式。" << std::endl;
				continue;
			}

			// 4. 配置并运行标定算法 (与 calibrate 函数逻辑一致)
			// struct CalibrateConfig {
			// 	bool hexgridfit = false;
			// 	bool autoEstimate = false;
			// 	bool genLUT = false;
			// 	bool saveLUT = false;
			// 	bool showPoints = true;
			// 	int diameter = 0;
			// 	int bitDepth = 8;
			// 	int views = 9;
			// 	int crop = 0;
			// 	float space = 1.0;
			// 	float rot = 0.0;
			// 	BayerPattern bayer = BayerPattern::NONE;
			// 	ExtractMethod ceMethod = ExtractMethod::Contour;
			// 	Orientation orientation = Orientation::HORZ;
			// };
			LFCalibrate calibrator;
			CalibrateConfig config;
			config.hexgridfit = true;
			config.autoEstimate = true;
			config.genLUT = false;
			config.saveLUT = false;
			config.showPoints = false;
			config.diameter = LYTRO_APPROX_PITCH;
			config.bitDepth = 10;
			config.views = 9;
			config.crop = 1;
			config.space = 1.0;
			config.rot = 0.0;
			config.bayer = BayerPattern::GRBG;
			config.ceMethod = ExtractMethod::CCA;
			config.orientation = Orientation::HORZ; // 对应 hex-col

			// 执行核心标定
			calibrator.run(img, config);

			// 获取标定结果的映射表
			auto maps = calibrator.getPoints();

			std::cout << "[Processed] 标定完成: " << fileName << " | 检出阵列大小: " << maps.first.cols << " x "
					  << maps.first.rows << std::endl;

			// 5. 将检测到的中心点保存为 JSON (格式严格与 evaluateAllMethods 兼容)
			json outputJson;

			// 构造元数据 (模拟合成数据集的 metadata)
			outputJson["metadata"] = {
				{"arrangement", arrangement},
				{"pitch", LYTRO_APPROX_PITCH}, // 这里保存初始预估值或你可以提取 calibrator 优化后的 pitch
				{"source", "Lytro Illum B5144000580"}};

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

			// 写入本地 JSON 文件
			std::string outJsonPath = saveDir + "/" + baseName + ".json";
			std::ofstream outF(outJsonPath);
			if (!outF.is_open()) {
				throw std::runtime_error("无法创建输出文件: " + outJsonPath);
			}
			outF << outputJson.dump(4); // 缩进 4 格
			outF.close();

		} catch (const std::exception &e) {
			std::cerr << "处理 " << fileName << " 时发生错误: " << e.what() << std::endl;
		}
	}

	std::cout << "==========================================" << std::endl;
	std::cout << "   真实数据集标定与 JSON 导出完成。" << std::endl;
	std::cout << "   保存路径: " << saveDir << std::endl;
	std::cout << "==========================================" << std::endl;
}
void test() {
	std::cout << cv::getBuildInformation() << std::endl;

	// 2. 检查当前运行时实际开启的核心数
	std::cout << "Threads: " << cv::getNumThreads() << std::endl;

	// 3. 检查特定指令集是否在当前环境被激活
	std::cout << "AVX2 Support: " << cv::checkHardwareSupport(CV_CPU_AVX2) << std::endl;
}
int main(int argc, char *argv[]) {
	if (argv[1] && std::string(argv[1]) == "calibrate") {
		calibrate();
	} else if (argv[1] && std::string(argv[1]) == "evaluate") {
		evaluateAllMethods();
	} else if (argv[1] && std::string(argv[1]) == "calibrate_lytro") {
		calibrateLytroRealData();
	} else {
		test();
	}

	return 0;
}
