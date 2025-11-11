#include "lfload.h"
#include "lfprocessor.h"
#include "window_base.h"

#include <QApplication>
#include <QDebug>
#include <QMetaMethod>
#include <QMetaObject>
#include <QObject>
#include <QPushButton>
#include <QThread>
#include <QtCore/qlogging.h>
#include <QtCore/qnamespace.h>
#include <QtCore/qobjectdefs.h>
#include <QtWidgets/qmainwindow.h>
#include <string>

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);
	WindowBase window;

	// std::string path(argv[1]);
	// bool		isRGB = strcmp(argv[2], "1") == 0 ? true : false;

	// window.worker		= new LFProcessor();
	// QThread *lfp_thread = new QThread();

	// window.worker->moveToThread(lfp_thread);
	// QObject::connect(lfp_thread, &QThread::started, window.worker,
	// 				 &LFProcessor::init_loader, Qt::QueuedConnection);
	// lfp_thread->start();

	// qDebug() << "Main thread:" << window.thread;
	// qDebug() << "LFProcessor thread:" << window.worker->thread();
	// qDebug() << "LFLoad thread:" << window.worker->loader->thread();

	// qDebug() << "Main threadId:" << QThread::currentThreadId();
	// QObject::connect(
	// 	window.button, &QPushButton::clicked, window.worker->loader,
	// 	[worker = window.worker, path, isRGB]() {
	// 		std::cout << "LFProcessor threadId: " << QThread::currentThreadId()
	// 				  << std::endl;
	// 		worker->loader->load(path, isRGB);
	// 	});
	// std::cout << "Main thread: " << QThread::currentThreadId() << std::endl;
	// QObject::connect(
	// 	window.button, &QPushButton::clicked, window.worker,
	// 	[&]() {
	// 		std::cout << "LFProcessor thread: " << QThread::currentThreadId()
	// 				  << std::endl;
	// 		window.worker->loader->invoke(&LFLoad::Core::load, path, isRGB);
	// 	},
	// 	Qt::QueuedConnection);
	// qDebug() << "当前线程：" << QThread::currentThread();
	// qDebug() << "lfp 所在线程：" << lfp->thread();
	// if (lfp->loader->thread()) {
	// 	qDebug() << "lfp->loader 所在线程：" << lfp->loader->thread();
	// }

	window.show();
	return app.exec();
}