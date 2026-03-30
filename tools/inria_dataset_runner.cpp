#include "colorequalize.h"
#include "json.hpp"
#include "lfcalibrate.h"
#include "lfio.h"
#include "lfisp.h"
#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
	const std::string input_dir = R"(F:\Light Field Dataset\LytroIllum_Dataset_INRIA_SIROCCO\Data)";
	const std::string calib_dir = R"(F:\Light Field Dataset\LytroIllum_Dataset_INRIA_SIROCCO\B5144000580)";
	const std::string output_dir = R"(F:\Light Field Dataset\Results\LightFieldLab)";
	const std::string log_csv = R"(F:\Light Field Dataset\Results\processing_time_log.csv)";

	auto isp = std::make_unique<LFIsp>();
	auto cal = std::make_unique<LFCalibrate>();

	CalibrateConfig caliConfig;
	caliConfig.hexgridfit = true;
	caliConfig.autoEstimate = true;
	caliConfig.genLUT = true;
	caliConfig.diameter = 0;
	caliConfig.bitDepth = 10;
	caliConfig.views = 9;
	caliConfig.crop = 2;
	caliConfig.space = 1.0f;
	caliConfig.rot = 0.0;
	caliConfig.bayer = BayerPattern::GRBG;
	caliConfig.ceMethod = ExtractMethod::GrayGravity;
	caliConfig.orientation = Orientation::HORZ;

	IspConfig ispConfig;
	ispConfig.dpcThreshold = 25;
	ispConfig.rawnr_sigma_s = 1.5f;
	ispConfig.rawnr_sigma_r = 15.0f;
	ispConfig.lscExp = 1.0f;
	ispConfig.uvnr_sigma_s = 1.5f;
	ispConfig.uvnr_sigma_r = 15.0f;
	ispConfig.enableRAW = true;
	ispConfig.enableBLC = true;
	ispConfig.enableDPC = true;
	ispConfig.enableRawNR = true;
	ispConfig.enableLSC = true;
	ispConfig.enableWB = true;
	ispConfig.enableDemosaic = true;
	ispConfig.enableExtract = true;
	ispConfig.enableDehex = true;
	ispConfig.enableCCM = true;
	ispConfig.enableGamma = true;
	ispConfig.enableCSC = false;
	ispConfig.enableUVNR = false;
	ispConfig.enableCE = false;
	ispConfig.enableSE = false;
	ispConfig.device = Device::GPU;
	ispConfig.benchmark = true;

	// ---------------------------------------------------------
	// 3. 循环遍历数据集
	// ---------------------------------------------------------
	if (!fs::exists(output_dir))
		fs::create_directories(output_dir);

	std::ofstream csvFile(log_csv);
	bool header_written = false; // [新增] 用于标记是否已写入动态表头
	if (csvFile.is_open()) {
		csvFile << "Filename,Calibration_ms,Processing_ms,ColorEq_ms,Total_Step_ms\n";
	}

	for (const auto &entry : fs::recursive_directory_iterator(input_dir)) {
		if (entry.is_regular_file() && entry.path().extension() == ".LFR") {
			std::string lfp_path = entry.path().string();
			std::string file_name = entry.path().stem().string();

			std::cout << std::format("Processing: {}", file_name) << std::endl;

			try {
				nlohmann::json meta;
				cv::Mat raw_img = LFIO::ReadLFP(lfp_path, meta);
				isp->parseJsonToConfig(meta, ispConfig);
				cv::Mat white;

				double time_cali = 0;
				double time_isp = 0;
				double time_color_eq = 0;
				double time_total_step = 0;

				Timer totalStepTimer;

				if (!raw_img.empty()) {
					json WhiteMeta;
					cv::Mat autoWhite = LFIO::ReadWhiteImageAuto(lfp_path, calib_dir, WhiteMeta);
					if (!autoWhite.empty()) {
						white = autoWhite; // 更新类的白图成员变量
						caliConfig.rot = WhiteMeta["master"]["picture"]["frameArray"][0]["frame"]["metadata"]["devices"]
												  ["mla"]["rotation"];
						isp->initConfig(white, ispConfig);
						// .set_white_gpu(white, ispConfig.black_level, ispConfig.white_level);
					}
					Timer caliTimer;

					cal->run(white, caliConfig);

					caliTimer.stop();
					time_cali = caliTimer.elapsed_ms();

					if (caliConfig.genLUT) {
						isp->maps.extract.clear();
						isp->maps.dehex.clear();
						isp->maps.extract = cal->getExtractMaps();
						isp->maps.dehex = cal->getDehexMaps();
						isp->update_resample_maps();
					}
				}

				isp->profiler_gpu.reset(); // 重置计数器与状态
				Timer ispTimer;

				// isp->set_lf_img(raw_img).process(ispConfig);
				// isp->set_lf_img(raw_img).process_fast(ispConfig);
				isp->set_lf_gpu(raw_img).process_gpu(ispConfig);

				ispTimer.stop();
				time_isp = ispTimer.elapsed_ms();
				isp->profiler_gpu.run_count = 1; // 确保单次运行逻辑正确

				Timer colorEqTimer;

				auto sai_data = isp->getSAIsGpu();
				// auto sai_data = isp->getSAIs();
				ColorEqualize::equalize(sai_data, ColorEqualizeMethod::HM_MKL_HM);
				colorEqTimer.stop();
				time_color_eq = colorEqTimer.elapsed_ms();

				totalStepTimer.stop();
				time_total_step = totalStepTimer.elapsed_ms();

				// if (csvFile.is_open()) {
				// 	csvFile << std::format("{},{:.2f},{:.2f},{:.2f},{:.2f}\n", file_name, time_cali, time_isp,
				// 						   time_color_eq, time_total_step);
				// 	csvFile.flush(); // 确保每一行处理完都实时写入硬盘
				// }
				// === [修改] 动态写入包含细化 ISP 模块的 CSV ===
				if (csvFile.is_open()) {
					// 1. 如果是第一张图，根据 Profiler 的项动态生成并写入表头
					if (!header_written) {
						csvFile << "Filename,Calibration_ms,ColorEq_ms,Total_Step_ms,ISP_Total_ms";
						for (const auto &item : isp->profiler_gpu.stats) {
							if (item.first != " Total Process") { // 剔除重复的总时间
								// 移除字符串首尾的空格以作标准列名
								std::string col_name = item.first;
								col_name.erase(0, col_name.find_first_not_of(" "));
								col_name.erase(col_name.find_last_not_of(" ") + 1);
								csvFile << ",ISP_" << col_name << "_ms";
							}
						}
						csvFile << "\n";
						header_written = true;
					}

					// 2. 写入基础宏观数据
					csvFile << std::format("{},{:.2f},{:.2f},{:.2f},{:.2f}", file_name, time_cali, time_color_eq,
										   time_total_step, time_isp);

					// 3. 遍历并写入细化的 ISP 模块耗时
					for (const auto &item : isp->profiler_gpu.stats) {
						if (item.first != " Total Process") {
							csvFile << std::format(",{:.3f}", item.second);
						}
					}
					csvFile << "\n";
					csvFile.flush();
				}
				// ==============================================

				// 后续保存逻辑

				std::string save_path = output_dir + "/" + file_name;
				if (!fs::exists(save_path))
					fs::create_directories(save_path);
				LFIO::SaveSAI(save_path, sai_data);

				std::cout << std::format("Done. Cali: {:.2f}ms, ISP: {:.2f}ms, ColorEq: {:.2f}ms, Total: {:.2f}ms",
										 time_cali, time_isp, time_color_eq, time_total_step)
						  << std::endl;
			} catch (const std::exception &e) {
				std::cerr << std::format("Error processing {}: {}", file_name, e.what()) << std::endl;
			}
		}
	}

	std::cout << "Batch processing complete." << std::endl;

	return 0;
}