#include "lfdata.h"
#include "lfload.h"
#include "window_base.h"

#include <QApplication>
#include <QFutureWatcher>
#include <QMainWindow>
#include <QObject>
#include <QString>
#include <QThread>
#include <QtWidgets/qpushbutton.h>
#include <cstring>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
int test_signals(int argc, char *argv[]) {
	QApplication app(argc, argv);
	WindowBase window;

	std::cout << "test LFLoad with Qt signals!" << std::endl;
	std::cout << "Main thread: " << QThread::currentThreadId() << std::endl;

	LFLoad::Worker *lfloader = new LFLoad::Worker();
	QString path(argv[1]);
	// std::string		  path(argv[1]);
	bool isRGB = strcmp(argv[2], "1") == 0 ? true : false;

	lfloader->moveToThread(window.thread);
	QObject::connect(window.thread, &QThread::finished, lfloader,
					 &LFLoad::Worker::deleteLater, Qt::QueuedConnection);
	QObject::connect(
		window.thread, &QThread::started, lfloader,
		[&]() { lfloader->load(QString(path), isRGB); }, Qt::QueuedConnection);
	window.thread->start();

	QObject::connect(window.button, &QPushButton::clicked, lfloader,
					 &LFLoad::Worker::getLF, Qt::QueuedConnection);
	QObject::connect(
		window.button, &QPushButton::clicked, lfloader,
		[&]() { lfloader->getLF(); }, Qt::QueuedConnection);
	QObject::connect(
		lfloader, &LFLoad::Worker::lfUpdated, &window,
		[&](const LightFieldPtr &ptr) {
			cv::imshow("image", ptr->getCenter());
			cv::waitKey();
		},
		Qt::QueuedConnection);

	QObject::connect(
		&window, &WindowBase::destroyed, &window,
		[&]() {
			window.thread->quit(); // 确保线程退出
			window.thread->wait(); // 等待线程结束
		},
		Qt::QueuedConnection);

	window.show();
	return app.exec();
}
// int test_template(int argc, char* argv[]) {
// 	QApplication app(argc, argv);
// 	WindowBase	 window;
// 	std::cout << "test LFLoad with template!" << std::endl;
// 	std::cout << "Main thread: " << QThread::currentThreadId() << std::endl;

// 	LFLoad::Worker* lfloader = new LFLoad::Worker(); // 不指定父对象
// 	std::string		  path(argv[1]);
// 	bool			  isRGB = strcmp(argv[2], "1") == 0 ? true : false;

// 	lfloader->moveToThread(window.thread);

// 	QObject::connect(
// 		window.thread, &QThread::started, lfloader,
// 		[&]() { lfloader->invoke(&LFLoad::Core::load, path, isRGB); },
// 		Qt::QueuedConnection);
// 	window.thread->start();

// 	// 在类中添加成员变量
// 	QFutureWatcher<LightField> futureWatcher;

// 	// 连接信号槽
// 	QObject::connect(&futureWatcher, &QFutureWatcher<LightField>::finished,
// 					 &window, [&]() {
// 						 auto result = futureWatcher.result();
// 						 cv::imshow("center", result.getCenter());
// 						 futureWatcher.waitForFinished();
// 					 });

// 	// 按钮点击处理
// 	QObject::connect(window.button, &QPushButton::clicked, &window,
// 					 [lfloader, &futureWatcher]() {
// 						 auto future = lfloader->invoke(&LFLoad::Core::getLF);
// 						 futureWatcher.setFuture(future);
// 					 });

// 	QObject::connect(window.thread, &QThread::finished, lfloader,
// 					 &LFLoad::Worker_signals::deleteLater,
// 					 Qt::QueuedConnection);
// 	QObject::connect(
// 		&window, &WindowBase::destroyed, &window,
// 		[&]() {
// 			window.thread->quit(); // 确保线程退出
// 			window.thread->wait(); // 等待线程结束
// 		},
// 		Qt::QueuedConnection);
// 	window.show();
// 	return app.exec();
// }

int main(int argc, char *argv[]) {
	int value;
	if (strcmp(argv[3], "0") == 0) {
		value = test_signals(argc, argv);
	} else {
		// value = test_template(argc, argv);
	}

	return value;
}