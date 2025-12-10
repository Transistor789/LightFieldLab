#ifndef CONFIG_H
#define CONFIG_H

#include <json.hpp>
#include <string>

// 使用 nlohmann::json 作为核心类型
using json = nlohmann::json;

class Config {
public:
	// --- 1. 单例访问点 (Meyers' Singleton) ---
	// C++11 保证静态局部变量初始化的线程安全
	static Config &Get() {
		static Config instance;
		return instance;
	}

	// --- 2. 禁止拷贝和赋值 (防止意外复制单例) ---
	Config(const Config &) = delete;
	Config &operator=(const Config &) = delete;

	// --- 3. 访问器 (Accessors) ---
	// 提供 const 和非 const 版本，灵活应对读取和修改需求

	// 程序配置 (Application Config)
	const json &app_cfg() const { return app_cfg_; }
	json &app_cfg() { return app_cfg_; }

	// 标定数据 (Calibrations)
	const json &calibs() const { return calibs_; }
	json &calibs() { return calibs_; }

	// 图像元数据 (Image Metadata)
	const json &img_meta() const { return img_meta_; }
	json &img_meta() { return img_meta_; }

	// --- 5. 文件操作 ---
	// 加载默认路径或指定路径的参数
	void readParams(const std::string &filePath = "");
	// 保存当前参数到文件
	void saveParams(const std::string &filePath = "");
	// 重置为硬编码的默认值
	void defaultValues();
	// 清空当前值
	void resetValues();

	// --- 7. 配置键常量 (Public Constants) ---
	// 保持为 public static，方便外部直接 Config::LFP_PATH 使用
	static inline constexpr const char *LFP_PATH = "lfp_path";
	static inline constexpr const char *CAL_PATH = "cal_path";
	static inline constexpr const char *CAL_META = "cal_meta";
	static inline constexpr const char *CAL_METH = "cal_meth";
	static inline constexpr const char *SMP_METH = "smp_meth";
	static inline constexpr const char *PTC_LENG = "ptc_leng";
	static inline constexpr const char *RAN_REFO = "ran_refo";

	static inline constexpr const char *OPT_CALI = "opt_cali";
	static inline constexpr const char *OPT_VIGN = "opt_vign";
	static inline constexpr const char *OPT_LIER = "opt_lier";
	static inline constexpr const char *OPT_CONT = "opt_cont";
	static inline constexpr const char *OPT_COLO = "opt_colo";
	static inline constexpr const char *OPT_AWB_ = "opt_awb_";
	static inline constexpr const char *OPT_SAT_ = "opt_sat_";
	static inline constexpr const char *OPT_VIEW = "opt_view";
	static inline constexpr const char *OPT_REFO = "opt_refo";
	static inline constexpr const char *OPT_REFI = "opt_refi";
	static inline constexpr const char *OPT_PFLU = "opt_pflu";
	static inline constexpr const char *OPT_ARTI = "opt_arti";
	static inline constexpr const char *OPT_ROTA = "opt_rota";
	static inline constexpr const char *OPT_DBUG = "opt_dbug";
	static inline constexpr const char *OPT_PRNT = "opt_prnt";
	static inline constexpr const char *OPT_DPTH = "opt_dpth";
	static inline constexpr const char *DIR_REMO = "dir_remo";

	// CALIBS_KEYS
	static inline constexpr const char *PAT_TYPE = "pat_type";
	static inline constexpr const char *PTC_MEAN = "ptc_mean";
	static inline constexpr const char *MIC_LIST = "mic_list";

private:
	// --- 构造函数私有化 ---
	Config();
	~Config() = default;

	// --- 内部数据成员 ---
	json app_cfg_;
	json calibs_;
	json img_meta_;

	std::string defaultConfigFile_;

	// 辅助函数
	static void ensureDirExists(const std::string &dirPath);
};

#endif // CONFIG_H