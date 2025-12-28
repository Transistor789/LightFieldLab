#include "mainwindow.h"

#include "qlogger.h"
#include "ui_mainwindow.h"
#include "widgetlogger.h"

#include <QSplitter> // 【核心】
#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logger.hpp>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent),
	  ui(new Ui::MainWindow),
	  ctrl(std::make_unique<LFControl>()) {
	ui->setupUi(this);

	// 连接日志
	connect(&QLogger::instance(), &QLogger::newLog, ui->widgetLog,
			&WidgetLogger::appendLog);

	connect(ui->widgetControl, &WidgetControl::requestLoadLFP, this,
			[this](const QString &path) { ctrl->read_img(path, false); });
	connect(ui->widgetControl, &WidgetControl::requestLoadWhite, this,
			[this](const QString &path) { ctrl->read_img(path, true); });
	connect(ui->widgetControl, &WidgetControl::requestLoadSAI, this,
			[this](const QString &path) { ctrl->read_sai(path); });

	connect(ctrl.get(), &LFControl::imageReady, ui->widgetImage,
			&WidgetImage::updateImage);

	// 启动业务

	ctrl->read_sai("data/bedroom");
	ctrl->read_img("data/toy.lfr", false);
	ctrl->read_img("data/MOD_0015.RAW", true);
}

MainWindow::~MainWindow() { delete ui; }
