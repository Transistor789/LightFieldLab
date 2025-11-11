#include "window_base.h"

#include <QApplication>
#include <QObject>
#include <QThread>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qpushbutton.h>
#include <iostream>

int main(int agrc, char *argv[]) {
	QApplication app(agrc, argv);
	WindowBase window;
	std::cout << "Main in thread: " << QThread::currentThread() << std::endl;
	QObject::connect(window.button, &QPushButton::clicked, &window, [&]() {
		std::cout << "Button clicked in thread: " << QThread::currentThread()
				  << std::endl;
	});
	window.show();
	return app.exec();
}