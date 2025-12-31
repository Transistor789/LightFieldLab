#include "widgetcontrol.h"

#include "dialogccm.h"
#include "dialogwbgains.h"
#include "lfparams.h"
#include "lfsr.h"
#include "logger.h"
#include "ui_widgetcontrol.h"

#include <QMenu>
#include <format>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qtmetamacros.h>

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
			params_->source.pathLFP = path.toStdString();
			emit requestLoadLFP(path);
		}
	});
	menuOpenLFP->addAction("子孔径", this, [this] { // 1. 处理 UI 逻辑（弹窗）
		QString path = QFileDialog::getExistingDirectory(
			this, "打开子孔径图像", "",
			QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

		if (!path.isEmpty()) {
			params_->source.pathSAI = path.toStdString();
			emit requestLoadSAI(path);
		}
	});
	ui->toolButtonLF->setMenu(menuOpenLFP);

	// 加载白图像
	connect(ui->toolButtonWhite, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(
			this, "打开白图像", "",
			"Lytro Files (*.lfp *.lfr *.raw);;Images (*.png *.bmp *jpeg "
			"*.jpg)");

		if (!path.isEmpty()) {
			params_->source.pathWhite = path.toStdString();
			emit requestLoadWhite(path);
		}
	});

	// 加载提取子孔径LUT
	connect(ui->toolButtonExtract, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开子孔径提取表",
													"", "LUT (*.bin)");

		if (!path.isEmpty()) {
			params_->source.pathExtract = path.toStdString();
			emit requestLoadExtractLUT(path);
		}
	});
	// 加载六边形重采样LUT
	connect(ui->toolButtonDehex, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开Dehex表", "",
													"LUT (*.bin)");

		if (!path.isEmpty()) {
			params_->source.pathDehex = path.toStdString();
			emit requestLoadDehexLUT(path);
		}
	});

	// 标定
	connect(ui->btnCalibrate, &QPushButton::clicked, this,
			&WidgetControl::requestCalibrate);
	// 生成LUT
	connect(ui->btnGenLUT, &QPushButton::clicked, this,
			&WidgetControl::requestGenLUT);

	// SAI

	// awb
	connect(ui->btnSetWBGains, &QPushButton::clicked, this, [this] {
		DialogWBGains dialog(this);
		dialog.setupParams(&params_->source);
		if (dialog.exec() == QDialog::Accepted) {
			updateUI();
		}
	});

	// ccm
	connect(ui->btnSetCCM, &QPushButton::clicked, this, [this] {
		DialogCCM dialog(this);
		dialog.setupParams(&params_->source);
		if (dialog.exec() == QDialog::Accepted) {
			updateUI();
		}
	});

	connect(ui->btnFastPreview, &QPushButton::clicked, this,
			&WidgetControl::requestFastPreview);

	connect(ui->btnISP, &QPushButton::clicked, this,
			&WidgetControl::requestISP);

	// 重聚焦
	connect(ui->btnRefocus, &QPushButton::clicked, this,
			&WidgetControl::requestRefocus);

	connect(ui->btnSR, &QPushButton::clicked, this, &WidgetControl::requestSR);
	connect(ui->btnDE, &QPushButton::clicked, this, &WidgetControl::requestDE);
}

WidgetControl::~WidgetControl() { delete ui; }

void WidgetControl::setupParams(LFParams *params) {
	params_ = params;
	if (params_ == nullptr) {
		return;
	}

	// Info
	connect(ui->comboBoxBayer, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->source.bayer = static_cast<BayerPattern>(index);
			});
	connect(ui->comboBoxBit, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->source.bitDepth = 8 + 2 * index; });

	// Calibrate
	connect(ui->comboBoxCCA, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->calibrate.useCCA = index; });
	connect(ui->checkBoxGridFit, &QCheckBox::toggled, this,
			[this](bool value) { params_->calibrate.gridfit = value; });
	connect(ui->checkBoxSaveLUT, &QCheckBox::toggled, this,
			[this](bool value) { params_->calibrate.saveLUT = value; });
	connect(ui->spinBoxLUTViews, &QSpinBox::valueChanged, this,
			[this](int value) { params_->calibrate.views = value; });

	// ISP
	connect(ui->checkBoxDPC, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableDPC = value; });
	connect(ui->checkBoxBLC, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableBLC = value; });
	connect(ui->checkBoxLSC, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableLSC = value; });
	connect(ui->checkBoxWB, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableAWB = value; });
	connect(ui->checkBoxDemosaic, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableDemosaic = value; });
	connect(ui->checkBoxCCM, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableCCM = value; });
	connect(ui->checkBoxGamma, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableGamma = value; });
	connect(ui->checkBoxExtract, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableExtract = value; });
	connect(ui->checkBoxDehex, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableDehex = value; });
	connect(ui->checkBoxColorEq, &QCheckBox::toggled, this,
			[this](bool value) { params_->isp.enableColorEq = value; });

	// SAI
	connect(ui->spinBoxHorz, &QSpinBox::valueChanged, this, [this](int value) {
		params_->sai.col = value;
		emit requestSAI(params_->sai.row, value);
	});
	connect(ui->spinBoxVert, &QSpinBox::valueChanged, this, [this](int value) {
		params_->sai.row = value;
		emit requestSAI(value, params_->sai.col);
	});
	connect(ui->pushButtonViewPlay, &QPushButton::toggled, this,
			[this](bool value) {
				params_->sai.isPlaying = value;

				if (value) {
					emit requestPlay();
					ui->pushButtonViewPlay->setText("停止");
				} else {
					ui->pushButtonViewPlay->setText("播放");
				}
			});
	connect(ui->pushButtonViewCenter, &QPushButton::clicked, this, [this] {
		emit requestSAI((params_->sai.row + 1) / 2, (params_->sai.col + 1) / 2);
	});

	// Refocus
	connect(ui->spinBoxRefocusCrop, &QSpinBox::valueChanged, this,
			[this](int value) { params_->refocus.crop = value; });
	connect(ui->doubleSpinBoxRefocusAlpha, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->refocus.alpha = value; });

	// SR
	connect(ui->comboBoxSRAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->sr.type = static_cast<LFParamsSR::Type>(index);
			});
	connect(ui->comboBoxSRScale, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->sr.scale = index + 2; });
	connect(
		ui->comboBoxSRPatchSize, &QComboBox::currentIndexChanged, this,
		[this](int index) { params_->sr.patchSize = index == 0 ? 128 : 196; });
	connect(ui->comboBoxSRViews, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->sr.views = 5 + 2 * index; });

	// DE
	connect(ui->comboBoxDepthAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->de.type = static_cast<LFParamsDE::Type>(index);
			});
	connect(ui->comboBoxDepthPatchColor, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->de.color = static_cast<LFParamsDE::Color>(index);
			});
	connect(
		ui->comboBoxDepthPatchSize, &QComboBox::currentIndexChanged, this,
		[this](int index) { params_->de.patchSize = index == 0 ? 128 : 196; });
	connect(ui->comboBoxDepthViews, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->de.views = params_->sr.views = 5 + 2 * index;
			});
}

void WidgetControl::updateUI() {
	if (params_ == nullptr) {
		return;
	}

	// 信息
	setValSilent(ui->comboBoxBayer, params_->source.bayer);
	setValSilent(ui->comboBoxBit, (params_->source.bitDepth - 8) / 2);
	setValSilent(ui->labelResValue, std::format("{}x{}", params_->source.width,
												params_->source.height));
	// 加载
	setValSilent(ui->lineEditLFP, params_->source.pathLFP);
	setValSilent(ui->lineEditWhite, params_->source.pathWhite);
	setValSilent(ui->lineEditExtract, params_->source.pathExtract);
	setValSilent(ui->lineEditDehex, params_->source.pathDehex);

	// ISP
	setValSilent(ui->checkBoxDPC, params_->isp.enableDPC);
	setValSilent(ui->checkBoxBLC, params_->isp.enableBLC);
	setValSilent(ui->checkBoxLSC, params_->isp.enableLSC);
	setValSilent(ui->checkBoxWB, params_->isp.enableAWB);
	setValSilent(ui->checkBoxDemosaic, params_->isp.enableDemosaic);
	setValSilent(ui->checkBoxCCM, params_->isp.enableCCM);
	setValSilent(ui->checkBoxGamma, params_->isp.enableGamma);
	setValSilent(ui->checkBoxExtract, params_->isp.enableExtract);
	setValSilent(ui->checkBoxDehex, params_->isp.enableDehex);
	setValSilent(ui->checkBoxColorEq, params_->isp.enableColorEq);

	// SAI
	setValSilent(ui->spinBoxHorz, params_->sai.col);
	setValSilent(ui->spinBoxVert, params_->sai.row);
	ui->spinBoxHorz->setMaximum(params_->sai.cols);
	ui->spinBoxVert->setMaximum(params_->sai.rows);

	// Refocus
	setValSilent(ui->spinBoxRefocusCrop, params_->refocus.crop);
	setValSilent(ui->doubleSpinBoxRefocusAlpha, params_->refocus.alpha);

	// SR
	setValSilent(ui->comboBoxSRAlgo, params_->sr.type);
	setValSilent(ui->comboBoxSRScale, params_->sr.scale - 2);
	setValSilent(ui->comboBoxSRPatchSize,
				 (params_->sr.patchSize == 128) ? 0 : 1);
	setValSilent(ui->comboBoxSRViews, (params_->sr.views - 5) / 2);

	// DE
	setValSilent(ui->comboBoxDepthAlgo, params_->de.type);
	setValSilent(ui->comboBoxDepthPatchColor,
				 static_cast<int>(params_->de.color));
	setValSilent(ui->comboBoxDepthPatchSize,
				 (params_->de.patchSize == 128) ? 0 : 1);
	setValSilent(ui->comboBoxDepthViews, (params_->de.views - 5) / 2);
}