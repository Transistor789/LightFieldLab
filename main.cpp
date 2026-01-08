#include "StreamRedirector.h"
#include "core/logger.h"
#include "mainwindow.h"
#include "ui/qlogger.h"

#include <QApplication>
#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logger.hpp>

void setupLogging() {
	// #ifdef QT_DEBUG
	// 劫持 std::cout -> 发给 Logger -> (CoreLogger处理格式) -> Adapter ->
	// UI
	static StreamRedirector redirectOut(std::cout, [](const std::string &msg) {
		// 直接调 Logger
		Logger::instance().log(LogLevel::Info, msg, "StdOut", 0);
	});

	static StreamRedirector redirectErr(std::cerr, [](const std::string &msg) {
		Logger::instance().log(LogLevel::Error, msg, "StdErr", 0);
	});
	// #else
	// cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	// #endif
}

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);

	QLogger::instance().init();
	setupLogging();

	MainWindow window;
	window.showMaximized();

	// 测试
	// LOG_INFO("Core Logger is ready!");
	// std::cout << "Std Cout is ready!" << std::endl;

	return app.exec();
}