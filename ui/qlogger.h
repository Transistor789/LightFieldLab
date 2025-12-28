#ifndef QLOGGER_H
#define QLOGGER_H

#include "logger.h" // 包含核心日志头

#include <QObject>

class QLogger : public QObject {
	Q_OBJECT
public:
	static QLogger &instance() {
		static QLogger inst;
		return inst;
	}

	// 初始化：把自己挂钩到 CoreLogger 上
	void init() {
		Logger::instance().setCallback(
			[this](LogLevel level, const std::string &msg) {
				// 类型转换
				int qtLevel = 0; // Info
				if (level == LogLevel::Warning)
					qtLevel = 1;
				else if (level == LogLevel::Error)
					qtLevel = 2;

				// 转换字符串并发送信号
				// 注意：因为回调可能来自工作线程，emit 是线程安全的
				emit newLog(qtLevel, QString::fromStdString(msg));
			});
	}

signals:
	void newLog(int level, QString msg);

private:
	QLogger() = default;
};

#endif // QTLOGADAPTER_H