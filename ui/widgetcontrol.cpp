#include "widgetcontrol.h"

#include "dialogccm.h"
#include "lfparams.h"
#include "ui_widgetcontrol.h"

#include <QMenu>
#include <format>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qcontainerfwd.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qtmetamacros.h>

WidgetControl::WidgetControl(QWidget *parent)
	: QWidget(parent), ui(new Ui::WidgetControl) {
	ui->setupUi(this);

	// =========================================================================
	// 1. 加载文件/文件夹
	// =========================================================================
	QMenu *menuOpenLFP = new QMenu(this);
	menuOpenLFP->addAction("原图", this, [this] {
		QString path = QFileDialog::getOpenFileName(
			this, "打开光场图像", "",
			"Lytro Files (*.lfp *.lfr *.raw);;Images (*.png *.bmp *jpeg "
			"*.jpg)");
		if (!path.isEmpty() && params_) {
			params_->path.lfp = path.toStdString();
			emit requestLoadLFP(path);
		}
	});
	menuOpenLFP->addAction("子孔径", this, [this] {
		QString path = QFileDialog::getExistingDirectory(
			this, "打开子孔径图像", "",
			QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
		if (!path.isEmpty() && params_) {
			params_->path.sai = path.toStdString();
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
		if (!path.isEmpty() && params_) {
			params_->path.white = path.toStdString();
			emit requestLoadWhite(path);
		}
	});

	// 加载 LUT
	connect(ui->toolButtonExtract, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开子孔径提取表",
													"", "LUT (*.bin)");
		if (!path.isEmpty() && params_) {
			params_->path.extractLUT = path.toStdString();
			emit requestLoadExtractLUT(path);
		}
	});
	connect(ui->toolButtonDehex, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开Dehex表", "",
													"LUT (*.bin)");
		if (!path.isEmpty() && params_) {
			params_->path.dehexLUT = path.toStdString();
			emit requestLoadDehexLUT(path);
		}
	});

	// =========================================================================
	// 2. 基础参数 (Info)
	// =========================================================================
	connect(ui->comboBoxBayer, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->isp.bayer = static_cast<BayerPattern>(index);
			});
	connect(ui->comboBoxBit, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->isp.bitDepth = 8 + 2 * index;
			});

	// =========================================================================
	// 3. 标定与预处理 (Calibration)
	// =========================================================================
	connect(ui->comboBoxCCA, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->calibrate.useCCA = index;
			});
	connect(ui->checkBoxGridFit, &QCheckBox::toggled, this, [this](bool value) {
		if (params_)
			params_->calibrate.gridFit = value;
	});
	connect(ui->checkBoxSaveLUT, &QCheckBox::toggled, this, [this](bool value) {
		if (params_)
			params_->calibrate.saveLUT = value;
	});
	connect(ui->spinBoxLUTViews, &QSpinBox::valueChanged, this,
			[this](int value) {
				if (params_)
					params_->calibrate.views = value;
			});

	// 按钮动作
	connect(ui->btnCalibrate, &QPushButton::clicked, this,
			&WidgetControl::requestCalibrate);
	connect(ui->btnGenLUT, &QPushButton::clicked, this,
			&WidgetControl::requestGenLUT);

	// =========================================================================
	// 4. ISP 管道控制
	// =========================================================================
	connect(ui->checkBoxDPC, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableDPC = val;
	});
	connect(ui->checkBoxBLC, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableBLC = val;
	});
	connect(ui->checkBoxLSC, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableLSC = val;
	});
	connect(ui->checkBoxWB, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableAWB = val;
	});

	connect(ui->checkBoxDemosaic, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableDemosaic = val;

		// 1. 控制直接隶属于 Demosaic 的子项
		ui->checkBoxCCM->setEnabled(val);
		ui->btnSetCCM->setEnabled(val);
		ui->checkBoxGamma->setEnabled(val);
		ui->doubleSpinBoxGamma->setEnabled(val);

		// 2. 控制中间层 Extract 的可用性
		ui->checkBoxExtract->setEnabled(val);

		bool subEnable = val && ui->checkBoxExtract->isChecked();

		ui->checkBoxDehex->setEnabled(subEnable);
		ui->checkBoxColorEq->setEnabled(subEnable);
		ui->comboBoxColorEq->setEnabled(subEnable);
	});

	connect(ui->checkBoxCCM, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableCCM = val;
	});
	connect(ui->checkBoxGamma, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableGamma = val;
	});

	connect(ui->checkBoxExtract, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableExtract = val;

		ui->checkBoxDehex->setEnabled(val);
		ui->checkBoxColorEq->setEnabled(val);
		ui->comboBoxColorEq->setEnabled(val);
	});
	connect(ui->checkBoxDehex, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableDehex = val;
	});
	connect(ui->checkBoxColorEq, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableColorEq = val;
	});

	connect(ui->comboBoxDPCAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->isp.dpcType = static_cast<LFParamsISP::DPCType>(index);
			});
	connect(ui->spinBoxThreshold, &QSpinBox::valueChanged, this,
			[this](int value) {
				params_->isp.dpcThreshold = static_cast<int>(value);
			});
	connect(ui->spinBoxBL, &QSpinBox::valueChanged, this, [this](int value) {
		params_->isp.black_level = static_cast<int>(value);
	});
	connect(ui->spinBoxWL, &QSpinBox::valueChanged, this, [this](int value) {
		params_->isp.white_level = static_cast<int>(value);
	});
	connect(ui->doubleSpinBoxExpo, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				params_->isp.lscExp = static_cast<float>(value);
			});
	connect(ui->doubleSpinBoxGain0, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				params_->isp.awb_gains[0] = static_cast<float>(value);
			});
	connect(ui->doubleSpinBoxGain1, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				params_->isp.awb_gains[1] = static_cast<float>(value);
			});
	connect(ui->doubleSpinBoxGain2, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				params_->isp.awb_gains[2] = static_cast<float>(value);
			});
	connect(ui->doubleSpinBoxGain3, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				params_->isp.awb_gains[3] = static_cast<float>(value);
			});
	connect(ui->comboBoxDemosaicAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->isp.demosaicType =
					static_cast<LFParamsISP::DemosaicType>(index);
			});
	connect(ui->btnSetCCM, &QPushButton::clicked, this, [this] {
		if (!params_)
			return;
		DialogCCM dialog(params_->isp.ccm_matrix, this);
		if (dialog.exec() == QDialog::Accepted) {
			updateUI();
		}
	});
	connect(ui->doubleSpinBoxGamma, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				params_->isp.gamma = static_cast<float>(value);
			});
	connect(ui->comboBoxColorEq, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->isp.colorEqType =
					static_cast<LFParamsISP::ColorEqType>(index);
			});
	connect(ui->btnFastPreview, &QPushButton::clicked, this,
			&WidgetControl::requestFastPreview);
	connect(ui->btnISP, &QPushButton::clicked, this,
			&WidgetControl::requestISP);

	// Dynamic
	connect(ui->pushButtonDetectCamera, &QPushButton::clicked, this,
			[this] { emit requestDetectCamera(); });
	connect(
		ui->pushButtonCapture, &QPushButton::toggled, this,
		[this](bool active) {
			params_->dynamic.isCapturing = active;
			ui->pushButtonCapture->setText(active ? "停止采集" : "开始采集");
			emit requestCapture(active);
			if (!active) {
				params_->dynamic.isProcessing = false;
				ui->pushButtonProcess->setText("开始处理");
				ui->pushButtonProcess->setChecked(false);
				emit requestProcess(false);
			}
		});
	connect(
		ui->pushButtonProcess, &QPushButton::toggled, this,
		[this](bool active) {
			params_->dynamic.isProcessing =
				active & params_->dynamic.isCapturing;
			ui->pushButtonProcess->setText(
				params_->dynamic.isProcessing ? "停止处理" : "开始处理");
			ui->pushButtonProcess->setChecked(params_->dynamic.isProcessing);
			emit requestProcess(params_->dynamic.isProcessing);
		});

	// =========================================================================
	// 5. 子孔径与播放 (SAI)
	// =========================================================================
	connect(ui->spinBoxHorz, &QSpinBox::valueChanged, this, [this](int value) {
		if (!params_)
			return;
		params_->sai.col = value;
		emit requestSAI(params_->sai.row, value);
	});
	connect(ui->spinBoxVert, &QSpinBox::valueChanged, this, [this](int value) {
		if (!params_)
			return;
		params_->sai.row = value;
		emit requestSAI(value, params_->sai.col);
	});
	// 【关键修复】播放按钮连接移到这里，只会连接一次
	connect(ui->pushButtonViewPlay, &QPushButton::toggled, this,
			[this](bool value) {
				if (!params_)
					return;
				params_->sai.isPlaying = value;
				ui->pushButtonViewPlay->setText(value ? "停止" : "播放");
				if (value) {
					emit requestPlay();
				}
			});
	connect(ui->pushButtonViewCenter, &QPushButton::clicked, this, [this] {
		if (!params_)
			return;
		params_->sai.row = (params_->sai.rows + 1) / 2;
		params_->sai.col = (params_->sai.cols + 1) / 2;
		emit requestSAI(params_->sai.row, params_->sai.col);
	});

	// =========================================================================
	// 6. 后处理 (Refocus, SR, DE)
	// =========================================================================
	// Refocus
	connect(ui->spinBoxRefocusCrop, &QSpinBox::valueChanged, this,
			[this](int value) {
				if (params_)
					params_->refocus.crop = value;
			});
	connect(ui->doubleSpinBoxRefocusAlpha, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				if (params_)
					params_->refocus.alpha = value;
			});
	connect(ui->btnRefocus, &QPushButton::clicked, this,
			&WidgetControl::requestRefocus);

	// SR
	connect(ui->comboBoxSRAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->sr.type = static_cast<LFParamsSR::Type>(index);
			});
	connect(ui->comboBoxSRScale, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->sr.scale = index + 2;
			});
	connect(ui->comboBoxSRPatchSize, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->sr.patchSize = index == 0 ? 128 : 196;
			});
	connect(ui->comboBoxSRViews, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->sr.views = 5 + 2 * index;
			});
	connect(ui->btnSR, &QPushButton::clicked, this, &WidgetControl::requestSR);

	// DE
	connect(ui->comboBoxDepthAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->de.type = static_cast<LFParamsDE::Type>(index);
			});
	connect(ui->comboBoxDepthPatchColor, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->de.color = static_cast<LFParamsDE::Color>(index);
			});
	connect(ui->comboBoxDepthPatchSize, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->de.patchSize = index == 0 ? 128 : 196;
			});
	connect(ui->comboBoxDepthViews, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->de.views = params_->sr.views = 5 + 2 * index;
			});
	connect(ui->btnDE, &QPushButton::clicked, this, &WidgetControl::requestDE);
}

WidgetControl::~WidgetControl() { delete ui; }

// setupParams 现在只负责设置指针和刷新 UI，不再进行信号连接
void WidgetControl::setupParams(LFParams *params) {
	params_ = params;
	if (params_) {
		updateUI();
	}
}

void WidgetControl::updateUI() {
	if (params_ == nullptr)
		return;

	// Info
	setValSilent(ui->comboBoxBayer, params_->isp.bayer);
	setValSilent(ui->comboBoxBit, (params_->isp.bitDepth - 8) / 2);
	setValSilent(ui->labelResValue,
				 std::format("{}x{}", params_->isp.width, params_->isp.height));

	// File Paths
	setValSilent(ui->lineEditLFP, params_->path.lfp);
	setValSilent(ui->lineEditWhite, params_->path.white);
	setValSilent(ui->lineEditExtract, params_->path.extractLUT);
	setValSilent(ui->lineEditDehex, params_->path.dehexLUT);

	// Static
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
	setValSilent(ui->comboBoxDPCAlgo, params_->isp.dpcType);
	setValSilent(ui->spinBoxThreshold, params_->isp.dpcThreshold);
	setValSilent(ui->spinBoxBL, params_->isp.black_level);
	setValSilent(ui->spinBoxWL, params_->isp.white_level);
	setValSilent(ui->doubleSpinBoxExpo, params_->isp.lscExp);
	setValSilent(ui->doubleSpinBoxGain0, params_->isp.awb_gains[0]);
	setValSilent(ui->doubleSpinBoxGain1, params_->isp.awb_gains[1]);
	setValSilent(ui->doubleSpinBoxGain2, params_->isp.awb_gains[2]);
	setValSilent(ui->doubleSpinBoxGain3, params_->isp.awb_gains[3]);
	setValSilent(ui->comboBoxDemosaicAlgo, params_->isp.demosaicType);
	setValSilent(ui->doubleSpinBoxGamma, params_->isp.gamma);
	setValSilent(ui->comboBoxColorEq, params_->isp.colorEqType);

	// Dynamic
	ui->spinBoxCamera->setEnabled(!params_->dynamic.cameraID.empty());
	if (!params_->dynamic.cameraID.empty()) {
		ui->labelCamera->setText(
			"检测到设备数: "
			+ QString::number(params_->dynamic.cameraID.size()));
	}

	// SAI
	ui->spinBoxHorz->setMaximum(params_->sai.cols);
	ui->spinBoxVert->setMaximum(params_->sai.rows);
	setValSilent(ui->spinBoxHorz, params_->sai.col);
	setValSilent(ui->spinBoxVert, params_->sai.row);
	// 播放按钮状态
	setValSilent(ui->pushButtonViewPlay, params_->sai.isPlaying);
	ui->pushButtonViewPlay->setText(params_->sai.isPlaying ? "停止" : "播放");

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