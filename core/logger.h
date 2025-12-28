#ifndef LOGGER_H
#define LOGGER_H

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

// 日志等级
enum class LogLevel { Info, Warning, Error, Debug };

// 日志回调函数签名：接收处理好的 格式化字符串 和 原始等级(用于UI着色)

class Logger {
public:
	using LogCallback =
		std::function<void(LogLevel level, const std::string &formattedMsg)>;

	static Logger &instance() {
		static Logger inst;
		return inst;
	}

	// 设置回调（Qt 适配器会调用这个）
	void setCallback(LogCallback cb) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_callback = cb;
	}

	// 核心 Log 函数
	void log(LogLevel level, const std::string &msg, const char *file,
			 int line) {
		std::lock_guard<std::mutex> lock(m_mutex);

		// 1. 获取时间 (纯 C++ 实现)
		auto now = std::chrono::system_clock::now();
		auto time_t_now = std::chrono::system_clock::to_time_t(now);
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					  now.time_since_epoch())
				  % 1000;

		std::stringstream ss;
		ss << std::put_time(std::localtime(&time_t_now), "[%H:%M:%S.")
		   << std::setfill('0') << std::setw(3) << ms.count() << "] ";

		// 2. 添加等级标签
		ss << levelToString(level) << " ";

		// 3. 添加文件行号 (仅 Debug 模式，利用宏控制)
#ifndef NDEBUG
		// 简易的文件名提取 (查找最后一个斜杠)
		std::string fileName = file;
		size_t lastSlash = fileName.find_last_of("/\\");
		if (lastSlash != std::string::npos)
			fileName = fileName.substr(lastSlash + 1);
		ss << "[" << fileName << ":" << line << "] ";
#endif

		// 4. 添加正文
		ss << msg;

		std::string finalStr = ss.str();

		// 5. 分发：如果有回调给回调，没回调打印到控制台
		if (m_callback) {
			m_callback(level, finalStr);
		} else {
			// 兜底输出
			std::cout << finalStr << std::endl;
		}

		// 6. (可选) 这里可以顺手写到 log.txt 文件里
		// saveToFile(finalStr);
	}

private:
	Logger() = default;

	std::string levelToString(LogLevel level) {
		switch (level) {
			case LogLevel::Info:
				return "[INFO]";
			case LogLevel::Warning:
				return "[WARN]";
			case LogLevel::Error:
				return "[ERROR]";
			case LogLevel::Debug:
				return "[DEBUG]";
			default:
				return "[UNK]";
		}
	}

	std::mutex m_mutex;
	LogCallback m_callback = nullptr;
};

// 宏定义
#define LOG_INFO(msg) \
	Logger::instance().log(LogLevel::Info, msg, __FILE__, __LINE__)
#define LOG_WARN(msg) \
	Logger::instance().log(LogLevel::Warning, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) \
	Logger::instance().log(LogLevel::Error, msg, __FILE__, __LINE__)

#endif // LOGGER_H