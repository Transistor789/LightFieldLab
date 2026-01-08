#ifndef LFIO_H
#define LFIO_H

#include "json.hpp"
#include "lfdata.h"

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

/**
 * @brief 光场图像 IO 工具类 (纯静态)
 * 负责文件的读取、写入和格式转换，不持有任何状态。
 */
class LFIO {
public:
	// 删除默认构造函数，防止实例化
	LFIO() = delete;

	// === 基础图像读取 ===
	static cv::Mat ReadStandardImage(const std::string &path);

	// === 光场图像读取 ===
	/**
	 * @brief 读取 LFP 文件并解析元数据
	 * @param path LFP 文件路径
	 * @param outMetadata [输出] 解析出的 JSON 元数据
	 */
	static cv::Mat ReadLFP(const std::string &path, json &outMetadata);

	/**
	 * @brief [方式一：自动] 根据 LFP 信息自动寻找并读取白图
	 * @param lfpPath LFP 文件路径 (用于提取查找键值)
	 * @param caliDir 标定库根目录
	 * @param outWhiteMeta [输出] 白图的元数据
	 * @return 修正后的白图
	 */
	static cv::Mat ReadWhiteImageAuto(const std::string &lfpPath,
									  const std::string &caliDir,
									  json &outWhiteMeta);

	/**
	 * @brief [方式二：手动] 指定路径读取白图 (新增)
	 * 适用于自动查找失败，或者用户手动选择文件的情况。
	 * 依然会自动寻找同名的 TXT/JSON 进行 BLC/AWB 处理。
	 * @param whitePath 白图文件的绝对路径 (通常是 .RAW)
	 * @param outWhiteMeta [输出] 白图的元数据
	 * @return 修正后的白图
	 */
	static cv::Mat ReadWhiteImageManual(const std::string &whitePath,
										json &outWhiteMeta);

	// === SAI 读取 ===
	static std::shared_ptr<LFData> ReadSAI(const std::string &path);

	// === LUT IO ===
	static bool SaveLookUpTables(const std::string &path,
								 const std::vector<cv::Mat> &maps, int winSize);
	static bool LoadLookUpTables(const std::string &path,
								 std::vector<cv::Mat> &maps, int &outWinSize);
};

#endif // LFIO_H