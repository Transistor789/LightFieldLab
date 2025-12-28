#include "widgetcontrol.h"

#include "dialogccm.h"
#include "dialogwbgains.h"
#include "logger.h"
#include "ui_widgetcontrol.h"

#include <QMenu>
#include <qpushbutton.h>

WidgetControl::WidgetControl(QWidget *parent)
	: QWidget(parent), ui(new Ui::WidgetControl) {
	ui->setupUi(this);

	QMenu *menuOpenLFP = new QMenu(this);
	menuOpenLFP->addAction("原图", this, [this] {
		QString path = QFileDialog::getOpenFileName(
			this, "打开光场图像", "",
			"Lytro Files (*.lfp *.lfr *.raw);;Images (*.png *.bmp *jpeg "
			"*.jpg)");

		if (!path.isEmpty()) {
			ui->lineEditLFP->setText(path);
			emit requestLoadLFP(path);
		}
	});
	menuOpenLFP->addAction("子孔径", this, [this] { // 1. 处理 UI 逻辑（弹窗）
		QString path = QFileDialog::getExistingDirectory(
			this, "打开子孔径图像", "",
			QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

		if (!path.isEmpty()) {
			ui->lineEditLFP->setText(path);
			emit requestLoadSAI(path);
		}
	});
	ui->toolButtonLF->setMenu(menuOpenLFP);

	connect(ui->toolButtonWhite, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(
			this, "打开白图像", "",
			"Lytro Files (*.lfp *.lfr *.raw);;Images (*.png *.bmp *jpeg "
			"*.jpg)");

		if (!path.isEmpty()) {
			ui->lineEditWhite->setText(path);
			emit requestLoadWhite(path);
		}
	});

	connect(ui->toolButtonSlice, &QToolButton::clicked, this, [this] {
		QString path =
			QFileDialog::getOpenFileName(this, "打开重排表", "", "LUT (*.bin)");

		if (!path.isEmpty()) {
			ui->lineEditWhite->setText(path);
			emit requestLoadSliceLUT(path);
		}
	});

	connect(ui->toolButtonDehex, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开Dehex表", "",
													"LUT (*.bin)");

		if (!path.isEmpty()) {
			ui->lineEditWhite->setText(path);
			emit requestLoadDehexLUT(path);
		}
	});

	connect(ui->btnSetWBGains, &QPushButton::clicked, this, [this] {
		DialogWBGains dialog(this);
		dialog.set({1, 1, 1, 1});
		LOG_INFO("setWBGains open");
		if (dialog.exec() == QDialog::Accepted) {
			LOG_INFO("setWBGains accepted");
		}
	});

	connect(ui->btnSetCCM, &QPushButton::clicked, this, [this] {
		DialogCCM dialog(this);
		LOG_INFO("setCCM open");
		if (dialog.exec() == QDialog::Accepted) {
			LOG_INFO("setCCM accepted");
		}
	});
}

WidgetControl::~WidgetControl() { delete ui; }
