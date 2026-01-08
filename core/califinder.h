#ifndef CALIFINDER_H
#define CALIFINDER_H

#include "json.hpp"

#include <string>


/**
 * @brief 标定文件查找器
 * 职责：根据相机序列号和几何引用ID，在指定目录中寻找匹配的白图像路径
 */
class CaliFinder {
public:
	/**
	 * @brief 构造函数
	 * @param cali_dir_root 标定数据根目录
	 */
	explicit CaliFinder(const std::string &cali_dir_root);

	/**
	 * @brief [核心接口] 根据元数据直接查找 (高性能，无文件IO)
	 * @param serial 相机序列号 (如 "B515210xxxxx")
	 * @param geo_ref 几何校正引用ID (Hash字符串)
	 * @return 找到的白图绝对路径，失败返回空字符串
	 */
	std::string findPath(const std::string &serial, const std::string &geo_ref);

	/**
	 * @brief [兼容接口] 给定LFP文件路径自动查找 (包含文件IO)
	 * @note 会重新读取LFP头部，性能较低，仅在没有预先解析JSON时使用
	 */
	std::string findWhiteImage(const std::string &lfp_path);

private:
	std::string _cali_root;

	// 内部辅助：读取LFP头（仅服务于兼容接口）
	nlohmann::json readLfpHeader(const std::string &path);

	// 内部辅助：解析 Manifest 寻找匹配
	std::string searchInManifest(const std::string &manifest_path,
								 const std::string &target_ref);
};

#endif // CALIFINDER_H