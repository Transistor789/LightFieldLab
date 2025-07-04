#include "lfload.h"
#include "lfrefocus.h"

#include <QApplication>
#include <QLabel>
#include <QMainWindow>
#include <QString>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QtCore/qtimer.h>
#include <QtCore/qvariant.h>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qboxlayout.h>
#include <QtWidgets/qlabel.h>
#include <QtWidgets/qpushbutton.h>
#include <chrono>
#include <cstring>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
// class Window : public QMainWindow {
// 	// Q_OBJECT
//    public:
// 	Window(int argc, char* argv[], QWidget* parent = nullptr)
// 		: QMainWindow(parent), counter(0) {
// 		std::string path  = std::string(argv[1]);
// 		bool		isRGB = strcmp(argv[2], "1") == 0 ? true : false;
// 		LFLoad	loader;
// 		loader.loadSAI(path, isRGB);
// 		std::vector<cv::Mat> LF_float32 = loader.getLF_float32();

// 		bool		  isGpu		= false;
// 		QLFRefocuser* refocuser = new QLFRefocuser(LF_float32, this);
// 		refocuser->execute("refocus", &LFRefocus::Core::setGPU, false);

// 		QWidget* centralWidget = new QWidget(this);
// 		setCentralWidget(centralWidget);

// 		QVBoxLayout* layout		   = new QVBoxLayout(centralWidget);
// 		QLabel*		 counterLabel  = new QLabel("Counter: 0", this);
// 		QLabel*		 refocusLabel  = new QLabel("Refocus time: 0", this);
// 		QPushButton* refocusButton = new QPushButton("Refocus", this);
// 		QHBoxLayout* hLayout	   = new QHBoxLayout(this);
// 		QLabel*		 gpuLable	   = new QLabel("Current use: CPU", this);
// 		QPushButton* gpuButton	   = new QPushButton("Switch", this);
// 		layout->addWidget(counterLabel);
// 		layout->addWidget(refocusButton);
// 		layout->addWidget(refocusLabel);
// 		hLayout->addWidget(gpuLable);
// 		hLayout->addWidget(gpuButton);
// 		layout->addLayout(hLayout);

// 		connect(refocuser, &QLFRefocuser::resultReady, this,
// 				[refocusLabel](QString op, QVariant result) {
// 					if (op == "refocus") {
// 						auto elapsed =
// 							result.value<std::chrono::duration<double>>();
// 						refocusLabel->setText(
// 							QString("Refocus time: %1").arg(elapsed.count()));
// 					}
// 				});
// 		connect(refocusButton, &QPushButton::clicked, this, [refocuser]() {
// 			// refocuser->execute(&LFRefocus::Worker::refocus, 1.5f, 2);
// 			refocuser->execute("refocus", 1.5f, 2);
// 		});
// 		connect(gpuButton, &QPushButton::clicked, this,
// 				[&isGpu, gpuLable, refocuser]() {
// 					isGpu = !isGpu;
// 					// refocuser->execute(&LFRefocus::Worker::setGpu, isGpu);
// 					refocuser->execute("setGpu", isGpu);
// 					gpuLable->setText(
// 						QString("Current use: %1").arg(isGpu ? "GPU" : "CPU"));
// 				});

// 		QTimer* timer = new QTimer(this);
// 		timer->start(100); // 每0.2秒更新一次
// 		connect(timer, &QTimer::timeout, this, [this, counterLabel]() {
// 			counter += 0.1;
// 			counterLabel->setText(QString("Counter: %1").arg(counter));
// 		});
// 	}

//    private:
// 	float counter = 0; // 计数器变量
// };
int main(int argc, char* argv[]) {
	std::cout << "Testing Refocus class" << std::endl;

	QApplication app(argc, argv);
	// Window		 window(argc, argv, nullptr);
	// window.show();
	return app.exec();
}