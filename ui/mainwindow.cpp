#include "mainwindow.h"

#include "qlogger.h"
#include "ui_mainwindow.h"
#include "widgetcontrol.h"
#include "widgetlogger.h"

#include <QSplitter> // 【核心】
#include <chrono>
#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <thread>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent),
	  ui(new Ui::MainWindow),
	  ctrl(std::make_unique<LFControl>()) {
	ui->setupUi(this);

	ui->widgetControl->setupParams(&ctrl->params);
	connect(ctrl.get(), &LFControl::paramsChanged, ui->widgetControl,
			&WidgetControl::updateUI);

	// 连接日志
	connect(&QLogger::instance(), &QLogger::newLog, ui->widgetLog,
			&WidgetLogger::appendLog);

	// 按键响应
	connect(ui->widgetControl, &WidgetControl::requestLoadSAI, this,
			[this](const QString &path) { ctrl->readSAI(path); });
	connect(ui->widgetControl, &WidgetControl::requestLoadLFP, this,
			[this](const QString &path) { ctrl->readImage(path, false); });
	connect(ui->widgetControl, &WidgetControl::requestLoadWhite, this,
			[this](const QString &path) { ctrl->readImage(path, true); });
	connect(ui->widgetControl, &WidgetControl::requestLoadExtractLUT, this,
			[this](const QString &path) { ctrl->readExtractLUT(path); });
	connect(ui->widgetControl, &WidgetControl::requestLoadDehexLUT, this,
			[this](const QString &path) { ctrl->readDehexLUT(path); });

	connect(ui->widgetControl, &WidgetControl::requestCalibrate, this,
			[this] { ctrl->calibrate(); });
	connect(ui->widgetControl, &WidgetControl::requestGenLUT, this,
			[this] { ctrl->genLUT(); });
	connect(ui->widgetControl, &WidgetControl::requestFastPreview, this,
			[this] { ctrl->fast_preview(); });
	connect(ui->widgetControl, &WidgetControl::requestISP, this,
			[this] { ctrl->process(); });
	// 重聚焦
	ui->widgetControl->setupParams(&ctrl->params);

	connect(ui->widgetControl, &WidgetControl::requestRefocus, this,
			[this] { ctrl->refocus(); });

	connect(ui->widgetControl, &WidgetControl::requestSR, this,
			[this] { ctrl->upsample(); });
	connect(ui->widgetControl, &WidgetControl::requestDE, this,
			[this] { ctrl->depth(); });

	// 图像就绪
	connect(ctrl.get(), &LFControl::imageReady, ui->widgetImage,
			&WidgetImage::updateImage);

// 启动业务
#ifdef NDEBUG
	ctrl->readExtractLUT("data/calibration/lut_extract_9.bin");
	ctrl->readDehexLUT("data/calibration/lut_dehex.bin");
	ctrl->readSAI("data/bedroom");
	ctrl->readImage("data/toy.lfr", false);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	ctrl->readImage("data/MOD_0015.RAW", true);
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	ctrl->calibrate();
	ctrl->refocus();
	ctrl->upsample();
	ctrl->depth();
#endif
}

MainWindow::~MainWindow() { delete ui; }
