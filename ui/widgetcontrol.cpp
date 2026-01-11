#include "widgetcontrol.h"

#include "centers_extract.h"
#include "colormatcher.h"
#include "dialogccm.h"
#include "lfdepth.h"
#include "lfisp.h"
#include "lfparams.h"
#include "lfsr.h"
#include "ui_widgetcontrol.h"

#include <QMenu>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qcontainerfwd.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qtmetamacros.h>

WidgetControl::WidgetControl(QWidget *parent)
	: QWidget(parent), ui(new Ui::WidgetControl) {
	ui->setupUi(this);

	// 1. 加载文件/文件夹
	QMenu *menuOpenLFP = new QMenu(this);
	menuOpenLFP->addAction("原图", this, [this] {
		QString path = QFileDialog::getOpenFileName(
			this, "打开光场图像", "",
			"*.lfp *.lfr *.raw *.png *.bmp *jpeg *.jpg");
		if (!path.isEmpty() && params_) {
			ui->lineEditLFP->setText(path);
			emit requestLoadLFP(path);
		}
	});
	menuOpenLFP->addAction("子孔径", this, [this] {
		QString path = QFileDialog::getExistingDirectory(
			this, "打开子孔径图像", "",
			QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
		if (!path.isEmpty() && params_) {
			ui->lineEditLFP->setText(path);
			emit requestLoadSAI(path);
		}
	});
	ui->toolButtonLF->setMenu(menuOpenLFP);

	// 加载白图像
	QMenu *menuOpenCali = new QMenu(this);
	menuOpenCali->addAction("文件夹", this, [this] {
		QString path = QFileDialog::getExistingDirectory(
			this, "打开标定数据所在文件夹", "",
			QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
		if (!path.isEmpty() && params_) {
			ui->lineEditWhite->setText(path);
		}
	});
	menuOpenCali->addAction("图像", this, [this] {
		QString path = QFileDialog::getOpenFileName(
			this, "打开白图像", "",
			"*.lfp *.lfr *.raw *.png *.bmp *jpeg *.jpg");
		if (!path.isEmpty() && params_) {
			ui->lineEditWhite->setText(path);
			emit requestLoadWhite(path);
		}
	});
	ui->toolButtonWhite->setMenu(menuOpenCali);

	// 加载 LUT
	connect(ui->toolButtonExtract, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开子孔径提取表",
													"", "LUT (*.bin)");
		if (!path.isEmpty() && params_) {
			ui->lineEditExtract->setText(path);
			emit requestLoadExtractLUT(path);
		}
	});
	connect(ui->toolButtonDehex, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开Dehex表", "",
													"LUT (*.bin)");
		if (!path.isEmpty() && params_) {
			ui->lineEditDehex->setText(path);
			emit requestLoadDehexLUT(path);
		}
	});

	// 2. 基础参数 (Info)

	connect(ui->comboBoxType, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->imageType = static_cast<ImageFileType>(index);
				ui->comboBoxBayer->setEnabled(index != 0);
				ui->comboBoxBit->setEnabled(index != 0);
				ui->lineEditHeight->setEnabled(index == 1);
				ui->lineEditWidth->setEnabled(index == 1);
			});
	connect(ui->comboBoxBayer, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->image.bayer = static_cast<BayerPattern>(index);
			});
	connect(ui->comboBoxBit, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->image.bitDepth = 8 + 2 * index;
			});
	connect(ui->lineEditHeight, &QLineEdit::textChanged, this,
			[this](QString str) {
				if (params_)
					params_->image.height = str.toInt();
			});
	connect(ui->lineEditWidth, &QLineEdit::textChanged, this,
			[this](QString str) {
				if (params_)
					params_->image.height = str.toInt();
			});

	// 3. 标定与预处理 (Calibration)

	// connect(ui->checkBoxCaliDemosaic, &QCheckBox::toggled, this,
	// 		[this](bool active) {
	// 			if (params_) {
	// 				params_->calibrate.demosaic = active;
	// 			}
	// 		});
	connect(ui->checkBoxDiameter, &QCheckBox::toggled, this,
			[this](bool active) {
				ui->spinBoxDiameter->setEnabled(!active);
				params_->calibrate.autoEstimate = active;
			});
	connect(ui->spinBoxDiameter, &QSpinBox::valueChanged, this,
			[this](int value) {
				if (params_)
					params_->calibrate.diameter = value;
			});
	connect(ui->comboBoxDetectAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->calibrate.ceMethod =
						static_cast<CentroidsExtract::Method>(index);
			});
	connect(ui->checkBoxGenLUT, &QCheckBox::toggled, this, [this](bool value) {
		if (params_)
			params_->calibrate.genLUT = value;
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

	// 4. ISP 管道控制

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
		ui->comboBoxColorEq->setEnabled(val);
	});
	connect(ui->checkBoxDehex, &QCheckBox::toggled, this, [this](bool val) {
		if (params_)
			params_->isp.enableDehex = val;
	});

	connect(ui->comboBoxDPCAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->isp.dpcMethod = static_cast<DpcMethod>(index);
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
				params_->isp.demosaicMethod =
					static_cast<DemosaicMethod>(index);
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
				params_->colorEqMethod =
					static_cast<ColorEqualizeMethod>(index);
			});
	connect(ui->comboBoxDevice, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				params_->isp.device = static_cast<Device>(index);
			});
	connect(ui->btnFastPreview, &QPushButton::clicked, this,
			&WidgetControl::requestFastPreview);
	connect(ui->btnISP, &QPushButton::clicked, this,
			&WidgetControl::requestISP);

	// Dynamic
	connect(ui->pushButtonDetectCamera, &QPushButton::clicked, this,
			[this] { emit requestDetectCamera(); });
	connect(ui->checkBoxShowLFP, &QCheckBox::toggled, this,
			[this](bool active) { params_->dynamic.showLFP = active; });
	connect(ui->checkBoxShowSAI, &QCheckBox::toggled, this,
			[this](bool active) { params_->dynamic.showSAI = active; });
	connect(
		ui->pushButtonCapture, &QPushButton::toggled, this,
		[this](bool active) {
			ui->pushButtonCapture->setText(active ? "停止采集" : "开始采集");
			emit requestCapture(active);
			if (!active) {
				if (ui->pushButtonProcess->isChecked()) {
					ui->pushButtonProcess->setChecked(false);
				} else {
					ui->pushButtonProcess->setText("开始处理");
				}
			}
		});
	connect(ui->pushButtonProcess, &QPushButton::toggled, this,
			[this](bool active) {
				bool canProcess = active;
				if (active && !ui->pushButtonCapture->isChecked()) {
					canProcess = false;
					{
						QSignalBlocker blocker(
							ui->pushButtonProcess); // 防止递归触发信号
						ui->pushButtonProcess->setChecked(false);
					}
				}
				ui->pushButtonProcess->setText(canProcess ? "停止处理"
														  : "开始处理");

				emit requestProcess(canProcess);
			});

	// 5. 子孔径与播放 (SAI)
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

	// 6. 后处理 (Refocus, SR, DE)

	// Color equalize
	connect(ui->pushButtonColorEq, &QPushButton::clicked, this,
			[this] { emit requestColorEq(); });
	// Refocus
	connect(ui->spinBoxRefocusCrop, &QSpinBox::valueChanged, this,
			[this](int value) {
				if (params_)
					params_->refocus.crop = value;
			});
	connect(ui->doubleSpinBoxRefocusShift, &QDoubleSpinBox::valueChanged, this,
			[this](double value) {
				if (params_)
					params_->refocus.shift = static_cast<float>(value);
			});
	connect(ui->btnRefocus, &QPushButton::clicked, this,
			&WidgetControl::requestRefocus);

	// SR
	connect(ui->comboBoxSRAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->sr.method = static_cast<SRMethod>(index);
			});
	connect(ui->comboBoxSRScale, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->sr.scale = index + 2;
			});
	connect(ui->btnSR, &QPushButton::clicked, this, &WidgetControl::requestSR);

	// DE
	connect(ui->comboBoxDepthAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->de.method = static_cast<DEMethod>(index);
			});
	connect(ui->comboBoxDepthPatchColor, &QComboBox::currentIndexChanged, this,
			[this](int index) {
				if (params_)
					params_->de.color = static_cast<LFParamsDE::Color>(index);
				emit requestChangingColor(index);
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

	// Path
	if (!params_->path.lfp.empty()) {
		setValSilent(ui->lineEditLFP, params_->path.lfp);
	} else {
		setValSilent(ui->lineEditLFP, params_->path.sai);
	}
	setValSilent(ui->lineEditWhite, params_->path.white);
	setValSilent(ui->lineEditExtract, params_->path.extractLUT);
	setValSilent(ui->lineEditDehex, params_->path.dehexLUT);

	// Info (修复数据源一致性)
	setValSilent(ui->comboBoxType, params_->imageType);

	// [修复] 从 isp.config 读取，确保与 ComboBox 的 set 信号源一致
	setValSilent(ui->comboBoxBayer, params_->isp.bayer);
	setValSilent(ui->comboBoxBit, (params_->isp.bitDepth - 8) / 2);

	setValSilent(ui->lineEditHeight, params_->image.height);
	setValSilent(ui->lineEditWidth, params_->image.width);

	// Calibrate (补全遗漏参数)
	ui->spinBoxDiameter->setEnabled(!params_->calibrate.autoEstimate);
	setValSilent(ui->spinBoxDiameter, params_->calibrate.diameter);
	setValSilent(ui->checkBoxDiameter, params_->calibrate.autoEstimate);
	setValSilent(ui->comboBoxDetectAlgo, params_->calibrate.ceMethod);
	setValSilent(ui->spinBoxLUTViews, params_->calibrate.views);
	setValSilent(ui->checkBoxSaveLUT, params_->calibrate.saveLUT);
	setValSilent(ui->checkBoxGenLUT, params_->calibrate.genLUT);

	// ISP - Static
	setValSilent(ui->checkBoxDPC, params_->isp.enableDPC);
	setValSilent(ui->checkBoxBLC, params_->isp.enableBLC);
	setValSilent(ui->checkBoxLSC, params_->isp.enableLSC);
	setValSilent(ui->checkBoxWB, params_->isp.enableAWB);
	setValSilent(ui->checkBoxDemosaic, params_->isp.enableDemosaic);
	setValSilent(ui->checkBoxCCM, params_->isp.enableCCM);
	setValSilent(ui->checkBoxGamma, params_->isp.enableGamma);
	setValSilent(ui->checkBoxExtract, params_->isp.enableExtract);
	setValSilent(ui->checkBoxDehex, params_->isp.enableDehex);

	setValSilent(ui->comboBoxDPCAlgo, params_->isp.dpcMethod);
	setValSilent(ui->spinBoxThreshold, params_->isp.dpcThreshold);
	setValSilent(ui->spinBoxBL, params_->isp.black_level);
	setValSilent(ui->spinBoxWL, params_->isp.white_level);
	setValSilent(ui->doubleSpinBoxExpo, params_->isp.lscExp);

	setValSilent(ui->doubleSpinBoxGain0, params_->isp.awb_gains[0]);
	setValSilent(ui->doubleSpinBoxGain1, params_->isp.awb_gains[1]);
	setValSilent(ui->doubleSpinBoxGain2, params_->isp.awb_gains[2]);
	setValSilent(ui->doubleSpinBoxGain3, params_->isp.awb_gains[3]);

	setValSilent(ui->comboBoxDemosaicAlgo, params_->isp.demosaicMethod);
	setValSilent(ui->doubleSpinBoxGamma, params_->isp.gamma);
	setValSilent(ui->comboBoxDevice, params_->isp.device);

	// ColorEq
	setValSilent(ui->comboBoxColorEq, params_->colorEqMethod);

	// Dynamic
	ui->spinBoxCamera->setEnabled(!params_->dynamic.cameraID.empty());
	if (!params_->dynamic.cameraID.empty()) {
		ui->labelCamera->setText(
			"检测到设备数: "
			+ QString::number(params_->dynamic.cameraID.size()));
	}
	setValSilent(ui->checkBoxShowLFP, params_->dynamic.showLFP);
	setValSilent(ui->checkBoxShowSAI, params_->dynamic.showSAI);

	// SAI
	ui->spinBoxHorz->setMaximum(params_->sai.cols);
	ui->spinBoxVert->setMaximum(params_->sai.rows);
	setValSilent(ui->spinBoxHorz, params_->sai.col);
	setValSilent(ui->spinBoxVert, params_->sai.row);

	setValSilent(ui->pushButtonViewPlay, params_->sai.isPlaying);
	ui->pushButtonViewPlay->setText(params_->sai.isPlaying ? "停止" : "播放");

	// Refocus
	setValSilent(ui->spinBoxRefocusCrop, params_->refocus.crop);
	setValSilent(ui->doubleSpinBoxRefocusShift, params_->refocus.shift);

	// SR
	setValSilent(ui->comboBoxSRAlgo, params_->sr.method);
	setValSilent(ui->comboBoxSRScale, params_->sr.scale - 2);

	// DE
	setValSilent(ui->comboBoxDepthAlgo, params_->de.method);
	setValSilent(ui->comboBoxDepthPatchColor,
				 static_cast<int>(params_->de.color));
}