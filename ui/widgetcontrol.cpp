#include "widgetcontrol.h"

#include "centers_extract.h"
#include "colormatcher.h"
#include "dialogccm.h"
#include "lfcalibrate.h"
#include "lfdepth.h"
#include "lfisp.h"
#include "lfparams.h"
#include "lfsr.h"
#include "ui_widgetcontrol.h"

#include <QMenu>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qcontainerfwd.h>
#include <qline.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qtmetamacros.h>
#include <qtoolbutton.h>
#include <sys/stat.h>

WidgetControl::WidgetControl(QWidget *parent) : QWidget(parent), ui(new Ui::WidgetControl) {
	ui->setupUi(this);

	// 光场原图
	connect(ui->lineEditLFP, &QLineEdit::textChanged, this,
			[this](QString str) { params_->path.lfp = str.toStdString(); });
	QMenu *menuOpenLFP = new QMenu(this);
	menuOpenLFP->addAction("原图", this, [this] {
		QString path =
			QFileDialog::getOpenFileName(this, "打开光场图像", "", "*.lfp *.lfr *.raw *.png *.bmp *jpeg *.jpg");
		if (!path.isEmpty() && params_) {
			params_->path.lfp = path.toStdString();
			ui->lineEditLFP->setText(path);
			emit requestLoadLFP(path);
		}
	});
	menuOpenLFP->addAction("子孔径", this, [this] {
		QString path = QFileDialog::getExistingDirectory(this, "打开子孔径图像", "",
														 QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
		if (!path.isEmpty() && params_) {
			params_->path.sai = path.toStdString();
			ui->lineEditLFP->setText(path);
			emit requestLoadSAI(path);
		}
	});
	ui->toolButtonLFBrowse->setMenu(menuOpenLFP);
	connect(ui->toolButtonLFPOpen, &QToolButton::clicked, this, [this] {
		QString path = ui->lineEditLFP->text();
		if (path.isEmpty() || !params_)
			return;
		QFileInfo checkInfo(path);
		if (checkInfo.exists()) {
			if (checkInfo.isFile()) {
				emit requestLoadLFP(path);
			} else if (checkInfo.isDir()) {
				emit requestLoadSAI(path);
			}
		}
	});

	// 标定白图
	connect(ui->lineEditWhite, &QLineEdit::textChanged, this,
			[this](QString str) { params_->path.white = str.toStdString(); });
	QMenu *menuOpenCali = new QMenu(this);
	menuOpenCali->addAction("文件夹", this, [this] {
		QString path = QFileDialog::getExistingDirectory(this, "打开标定数据所在文件夹", "",
														 QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
		if (!path.isEmpty() && params_) {
			params_->path.white = path.toStdString();
			ui->lineEditWhite->setText(path);
		}
	});
	menuOpenCali->addAction("图像", this, [this] {
		QString path =
			QFileDialog::getOpenFileName(this, "打开白图像", "", "*.lfp *.lfr *.raw *.png *.bmp *jpeg *.jpg");
		if (!path.isEmpty() && params_) {
			params_->path.white = path.toStdString();
			ui->lineEditWhite->setText(path);
			emit requestLoadWhite(path);
		}
	});
	ui->toolButtonWhiteBrowse->setMenu(menuOpenCali);
	connect(ui->toolButtonWhiteOpen, &QToolButton::clicked, this, [this] {
		QString path = ui->lineEditWhite->text();
		if (path.isEmpty() || !params_)
			return;

		QFileInfo checkInfo(path);
		if (checkInfo.exists()) {
			if (checkInfo.isFile()) {
				emit requestLoadWhite(path);
			}
		}
	});

	// 图像信息
	connect(ui->comboBoxType, &QComboBox::currentIndexChanged, this, [this](int index) {
		params_->imageType = static_cast<ImageFileType>(index);
		ui->comboBoxBayer->setEnabled(index != 0);
		ui->comboBoxBit->setEnabled(index != 0);
		ui->lineEditHeight->setEnabled(index == 1);
		ui->lineEditWidth->setEnabled(index == 1);
	});
	connect(ui->comboBoxBayer, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->image.bayer = static_cast<BayerPattern>(index); });
	connect(ui->comboBoxBit, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->image.bitDepth = 8 + 2 * index; });
	connect(ui->lineEditHeight, &QLineEdit::textChanged, this,
			[this](QString str) { params_->image.height = str.toInt(); });
	connect(ui->lineEditWidth, &QLineEdit::textChanged, this,
			[this](QString str) { params_->image.height = str.toInt(); });

	// 提取表
	connect(ui->lineEditExtract, &QLineEdit::textChanged, this,
			[this](QString str) { params_->path.extractLUT = str.toStdString(); });
	connect(ui->toolButtonExtractOpen, &QToolButton::clicked, this,
			[this] { emit requestLoadExtractLUT(QString::fromStdString(params_->path.extractLUT)); });
	connect(ui->toolButtonExtractBrowse, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开子孔径提取表", "", "LUT (*.bin)");
		if (!path.isEmpty() && params_) {
			params_->path.extractLUT = path.toStdString();
			ui->lineEditExtract->setText(path);
			emit requestLoadExtractLUT(path);
		}
	});

	// Dehex表
	connect(ui->lineEditDehex, &QLineEdit::textChanged, this,
			[this](QString str) { params_->path.dehexLUT = str.toStdString(); });
	connect(ui->toolButtonDehexOpen, &QToolButton::clicked, this,
			[this] { emit requestLoadDehexLUT(QString::fromStdString(params_->path.dehexLUT)); });
	connect(ui->toolButtonDehexBrowse, &QToolButton::clicked, this, [this] {
		QString path = QFileDialog::getOpenFileName(this, "打开Dehex表", "", "LUT (*.bin)");
		if (!path.isEmpty() && params_) {
			params_->path.dehexLUT = path.toStdString();
			ui->lineEditDehex->setText(path);
			emit requestLoadDehexLUT(path);
		}
	});

	connect(ui->comboBoxLayout, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->calibrate.orientation = static_cast<Orientation>(index); }); // 排列
	connect(ui->checkBoxDiameter, &QCheckBox::toggled, this, [this](bool active) {
		ui->spinBoxDiameter->setEnabled(!active);
		params_->calibrate.autoEstimate = active;
	}); // 微透镜尺寸
	connect(ui->spinBoxDiameter, &QSpinBox::valueChanged, this,
			[this](int value) { params_->calibrate.diameter = value; }); // 微透镜尺寸
	connect(ui->doubleSpinBoxSpace, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->calibrate.space = static_cast<float>(value); }); // 微透镜尺寸
	connect(ui->comboBoxDetectAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->calibrate.ceMethod = static_cast<ExtractMethod>(index); }); // 检测方法
	connect(ui->spinBoxLUTViews, &QSpinBox::valueChanged, this,
			[this](int value) { params_->calibrate.views = value; }); // 视角数
	connect(ui->checkBoxGenLUT, &QCheckBox::toggled, this,
			[this](bool value) { params_->calibrate.genLUT = value; }); // 生成查找表
	connect(ui->checkBoxHexGridFit, &QCheckBox::toggled, this,
			[this](bool value) { params_->calibrate.hexgridfit = value; }); // 保存查找表
	connect(ui->btnCalibrate, &QPushButton::clicked, this,
			&WidgetControl::requestCalibrate); // 标定按钮

	// 4. ISP 管道控制
	connect(ui->checkBoxBLC, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableBLC = val; });
	connect(ui->checkBoxDPC, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableDPC = val; });
	connect(ui->checkBoxNR, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableNR = val; });
	connect(ui->checkBoxLSC, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableLSC = val; });
	connect(ui->checkBoxWB, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableAWB = val; });
	connect(ui->checkBoxDemosaic, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableDemosaic = val; });
	connect(ui->checkBoxCCM, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableCCM = val; });
	connect(ui->checkBoxGamma, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableGamma = val; });
	connect(ui->checkBoxExtract, &QCheckBox::toggled, this, [this](bool val) {
		params_->isp.enableExtract = val;
		ui->checkBoxDehex->setEnabled(val);
	});
	connect(ui->checkBoxDehex, &QCheckBox::toggled, this, [this](bool val) { params_->isp.enableDehex = val; });

	connect(ui->spinBoxBL, &QSpinBox::valueChanged, this,
			[this](int value) { params_->isp.black_level = static_cast<int>(value); });
	connect(ui->spinBoxWL, &QSpinBox::valueChanged, this,
			[this](int value) { params_->isp.white_level = static_cast<int>(value); });
	connect(ui->spinBoxThreshold, &QSpinBox::valueChanged, this,
			[this](int value) { params_->isp.dpcThreshold = static_cast<int>(value); });
	connect(ui->doubleSpinBoxSigmaS, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.nr_sigma_s = static_cast<float>(value); });
	connect(ui->doubleSpinBoxSigmaR, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.nr_sigma_r = static_cast<float>(value); });
	connect(ui->doubleSpinBoxExpo, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.lscExp = static_cast<float>(value); });
	connect(ui->doubleSpinBoxGain0, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.awb_gains[0] = static_cast<float>(value); });
	connect(ui->doubleSpinBoxGain1, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.awb_gains[1] = static_cast<float>(value); });
	connect(ui->doubleSpinBoxGain2, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.awb_gains[2] = static_cast<float>(value); });
	connect(ui->doubleSpinBoxGain3, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.awb_gains[3] = static_cast<float>(value); });
	connect(ui->comboBoxDemosaicAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->isp.demosaicMethod = static_cast<DemosaicMethod>(index); });
	connect(ui->btnSetCCM, &QPushButton::clicked, this, [this] {
		DialogCCM dialog(params_->isp.ccm_matrix, this);
		if (dialog.exec() == QDialog::Accepted) {
			updateUI();
		}
	});
	connect(ui->doubleSpinBoxGamma, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->isp.gamma = static_cast<float>(value); });
	connect(ui->comboBoxColorEq, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->colorEqMethod = static_cast<ColorEqualizeMethod>(index); });
	connect(ui->comboBoxDevice, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->isp.device = static_cast<Device>(index); });
	connect(ui->btnFastPreview, &QPushButton::clicked, this, &WidgetControl::requestFastPreview);
	connect(ui->btnISP, &QPushButton::clicked, this, &WidgetControl::requestISP);

	// Dynamic
	connect(ui->pushButtonDetectCamera, &QPushButton::clicked, this, [this] { emit requestDetectCamera(); });
	connect(ui->checkBoxShowLFP, &QCheckBox::toggled, this, [this](bool active) { params_->dynamic.showLFP = active; });
	connect(ui->checkBoxShowSAI, &QCheckBox::toggled, this, [this](bool active) { params_->dynamic.showSAI = active; });
	connect(ui->pushButtonCapture, &QPushButton::toggled, this, [this](bool active) {
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
	connect(ui->pushButtonProcess, &QPushButton::toggled, this, [this](bool active) {
		bool canProcess = active;
		if (active && !ui->pushButtonCapture->isChecked()) {
			canProcess = false;
			{
				QSignalBlocker blocker(ui->pushButtonProcess); // 防止递归触发信号
				ui->pushButtonProcess->setChecked(false);
			}
		}
		ui->pushButtonProcess->setText(canProcess ? "停止处理" : "开始处理");

		emit requestProcess(canProcess);
	});
	connect(ui->pushButtonSaveSAI, &QPushButton::clicked, this, [this] {
		QString path = QFileDialog::getExistingDirectory(this, "保存子孔径图像", "",
														 QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
		if (!path.isEmpty()) {
			emit requestSaveSAI(path);
		}
	});

	// 5. 子孔径与播放 (SAI)
	connect(ui->spinBoxHorz, &QSpinBox::valueChanged, this, [this](int value) {
		params_->sai.col = value;
		emit requestSAI(params_->sai.row, value);
	});
	connect(ui->spinBoxVert, &QSpinBox::valueChanged, this, [this](int value) {
		params_->sai.row = value;
		emit requestSAI(value, params_->sai.col);
	});
	connect(ui->pushButtonViewPlay, &QPushButton::toggled, this, [this](bool value) {
		params_->sai.isPlaying = value;
		ui->pushButtonViewPlay->setText(value ? "停止" : "播放");
		if (value) {
			emit requestPlay();
		}
	});
	connect(ui->pushButtonViewCenter, &QPushButton::clicked, this, [this] {
		params_->sai.row = (params_->sai.rows + 1) / 2;
		params_->sai.col = (params_->sai.cols + 1) / 2;
		emit requestSAI(params_->sai.row, params_->sai.col);
	});

	// 6. 后处理 (Refocus, SR, DE)

	// Color equalize
	connect(ui->pushButtonColorEq, &QPushButton::clicked, this, [this] { emit requestColorEq(); });
	// Refocus
	connect(ui->spinBoxRefocusCrop, &QSpinBox::valueChanged, this,
			[this](int value) { params_->refocus.crop = value; });
	connect(ui->doubleSpinBoxRefocusShift, &QDoubleSpinBox::valueChanged, this,
			[this](double value) { params_->refocus.shift = static_cast<float>(value); });
	connect(ui->btnRefocus, &QPushButton::clicked, this, &WidgetControl::requestRefocus);

	// SR
	connect(ui->comboBoxSRAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->sr.method = static_cast<SRMethod>(index); }); // 算法选择
	connect(ui->comboBoxSRScale, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->sr.scale = index + 2; }); // 倍数
	connect(ui->btnSR, &QPushButton::clicked, this,
			&WidgetControl::requestSR); // 运行按钮

	// DE
	connect(ui->comboBoxDEAlgo, &QComboBox::currentIndexChanged, this,
			[this](int index) { params_->de.color = static_cast<DEColor>(index); });
	connect(ui->comboBoxDEColor, &QComboBox::currentIndexChanged, this, [this](int index) {
		params_->de.color = static_cast<DEColor>(index);
		emit requestChangingColor(index);
	}); // 改变颜色
	connect(ui->btnDE, &QToolButton::clicked, this,
			&WidgetControl::requestDE); // 运行按钮
}

WidgetControl::~WidgetControl() { delete ui; }

// setupParams 现在只负责设置指针和刷新 UI，不再进行信号连接
void WidgetControl::setupParams(LFParams *params) {
	params_ = params;
	updateUI();
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
	setValSilent(ui->comboBoxLayout, params_->calibrate.orientation);
	setValSilent(ui->spinBoxDiameter, params_->calibrate.diameter);
	setValSilent(ui->checkBoxDiameter, params_->calibrate.autoEstimate);
	setValSilent(ui->comboBoxDetectAlgo, params_->calibrate.ceMethod);
	setValSilent(ui->spinBoxLUTViews, params_->calibrate.views);
	setValSilent(ui->doubleSpinBoxSpace, params_->calibrate.space);
	setValSilent(ui->checkBoxHexGridFit, params_->calibrate.hexgridfit);
	setValSilent(ui->checkBoxGenLUT, params_->calibrate.genLUT);

	// ISP - Static
	setValSilent(ui->checkBoxBLC, params_->isp.enableBLC);
	setValSilent(ui->checkBoxDPC, params_->isp.enableDPC);
	setValSilent(ui->checkBoxNR, params_->isp.enableNR);
	setValSilent(ui->checkBoxLSC, params_->isp.enableLSC);
	setValSilent(ui->checkBoxWB, params_->isp.enableAWB);
	setValSilent(ui->checkBoxDemosaic, params_->isp.enableDemosaic);
	setValSilent(ui->checkBoxCCM, params_->isp.enableCCM);
	setValSilent(ui->checkBoxGamma, params_->isp.enableGamma);
	setValSilent(ui->checkBoxExtract, params_->isp.enableExtract);
	setValSilent(ui->checkBoxDehex, params_->isp.enableDehex);

	setValSilent(ui->spinBoxBL, params_->isp.black_level);
	setValSilent(ui->spinBoxWL, params_->isp.white_level);
	setValSilent(ui->spinBoxThreshold, params_->isp.dpcThreshold);
	setValSilent(ui->doubleSpinBoxSigmaS, params_->isp.nr_sigma_s);
	setValSilent(ui->doubleSpinBoxSigmaR, params_->isp.nr_sigma_r);
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
	if (!params_->dynamic.cameraID.empty()) {
		ui->labelCamera->setText(QString("设备ID: %1").arg(params_->dynamic.cameraID[0]));
	} else {
		ui->labelCamera->setText(QString("无可用设备"));
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
	setValSilent(ui->comboBoxDEAlgo, params_->de.method);
	setValSilent(ui->comboBoxDEColor, static_cast<int>(params_->de.color));
}