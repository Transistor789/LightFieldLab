#ifndef WINDOW_BASE_H
#define WINDOW_BASE_H
#include "lfprocessor.h"

#include <QCloseEvent>
#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QThread>
#include <QVBoxLayout>

// template <typename Worker>
class WindowBase : public QMainWindow {
	Q_OBJECT
public:
	WindowBase(QWidget* parent = nullptr) : QMainWindow(parent) {
		QWidget* centralWidget = new QWidget(this);
		QVBoxLayout* layout = new QVBoxLayout(centralWidget);
		button = new QPushButton("execute", this);

		layout->addWidget(button);
		centralWidget->setLayout(layout);
		setCentralWidget(centralWidget);

		thread = new QThread();
	}
	~WindowBase() {
		if (thread != nullptr) {
			thread->quit();
			thread->wait();
			delete thread;
			thread = nullptr;
		}
		delete worker;
	}
	QThread* thread;
	LFProcessor* worker;
	QPushButton* button;
};

#endif