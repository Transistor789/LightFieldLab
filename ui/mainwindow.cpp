#include "mainwindow.h"

#include "qlogger.h"
#include "ui_mainwindow.h"
#include "widgetcontrol.h"
#include "widgetlogger.h"

#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logger.hpp>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), ui(new Ui::MainWindow), ctrl(std::make_unique<LFControl>()) {
	ui->setupUi(this);

	initStatusBar();
	// 连接全量信息 (鼠标动的时候刷新一切)
	connect(ui->widgetImage, &WidgetImage::imageMouseMoved, this, &MainWindow::updateMouseInfo);

	// 连接缩放信息 (滚轮动的时候只刷新缩放)
	connect(ui->widgetImage, &WidgetImage::imageZoomChanged, this, [this](double scale) {
		int percent = static_cast<int>(scale * 100);
		lblZoom->setText(QString("%1%").arg(percent));
	});

	m_fpsTimer = new QTimer(this);
	m_fpsTimer->setInterval(1000);

	connect(m_fpsTimer, &QTimer::timeout, this, [this]() {
		// 1. 获取距离上一次触发经过了多少毫秒 (例如可能是 1002ms)
		qint64 elapsedMs = m_fpsTimeCalculator.restart();

		// 防止除以0的防御性编程
		if (elapsedMs == 0)
			elapsedMs = 1;

		// 2. 获取并清零计数器
		int countCap = ctrl->params.dynamic.capFrameCount.exchange(0);
		int countProc = ctrl->params.dynamic.procFrameCount.exchange(0);
		int countUI = m_uiFrameCount;
		m_uiFrameCount = 0;

		// 3. 计算精确帧率 ( 帧数 * 1000 / 实际毫秒数 )
		double fpsCap = countCap * 1000.0 / elapsedMs;
		double fpsProc = countProc * 1000.0 / elapsedMs;
		double fpsUI = countUI * 1000.0 / elapsedMs;

		// 4. 更新状态栏 (保留1位小数)
		// arg(double, fieldWidth, format, precision) -> 'f', 1
		// 表示浮点数保留1位
		lblFPS->setText(QString("FPS: %1 / %2 / %3 (Cap/Proc/UI)")
							.arg(fpsCap, 0, 'f', 1)
							.arg(fpsProc, 0, 'f', 1)
							.arg(fpsUI, 0, 'f', 1));

		// 5. 颜色警示逻辑 (使用计算出的 double 值判断)
		if (fpsCap < 10.0) {
			lblFPS->setStyleSheet("color: red; padding: 0 5px;");
		} else {
			lblFPS->setStyleSheet("color: green; padding: 0 5px;");
		}
	});
	m_fpsTimer->start();

	ui->widgetControl->setupParams(&ctrl->params);
	connect(ctrl.get(), &LFControl::paramsChanged, ui->widgetControl, &WidgetControl::updateUI);

	// 连接日志
	connect(&QLogger::instance(), &QLogger::newLog, ui->widgetLogger, &WidgetLogger::appendLog);

	// 按键响应
	connect(ui->widgetControl, &WidgetControl::requestLoadSAI, this,
			[this](const QString &path) { ctrl->readSAI(path); });
	connect(ui->widgetControl, &WidgetControl::requestLoadLFP, this,
			[this](const QString &path) { ctrl->readLFP(path); });
	connect(ui->widgetControl, &WidgetControl::requestLoadWhite, this,
			[this](const QString &path) { ctrl->readWhite(path); });
	connect(ui->widgetControl, &WidgetControl::requestLoadExtractLUT, this,
			[this](const QString &path) { ctrl->readExtractLUT(path); });
	connect(ui->widgetControl, &WidgetControl::requestLoadDehexLUT, this,
			[this](const QString &path) { ctrl->readDehexLUT(path); });

	connect(ui->widgetControl, &WidgetControl::requestCalibrate, this, [this] { ctrl->calibrate(); });
	connect(ui->widgetControl, &WidgetControl::requestFastPreview, this, [this] { ctrl->fast_preview(); });
	connect(ui->widgetControl, &WidgetControl::requestISP, this, [this] { ctrl->process(); });
	connect(ui->widgetControl, &WidgetControl::requestSaveSAI, this,
			[this](const QString &path) { ctrl->saveSAI(path); });
	connect(ui->widgetControl, &WidgetControl::requestDetectCamera, this, [this] { ctrl->detectCamera(); });
	connect(ui->widgetControl, &WidgetControl::requestCapture, this,
			[this](bool active) { ctrl->setCapturing(active); });
	connect(ui->widgetControl, &WidgetControl::requestProcess, this,
			[this](bool active) { ctrl->setProcessing(active); });
	connect(ui->widgetControl, &WidgetControl::requestPlay, this, [this] { ctrl->play(); });
	connect(ui->widgetControl, &WidgetControl::requestSAI, this,
			[this](int row, int col) { ctrl->updateSAI(row, col); });

	connect(ui->widgetControl, &WidgetControl::requestColorEq, this, [this] { ctrl->color_equalize(); });
	connect(ui->widgetControl, &WidgetControl::requestRefocus, this, [this] { ctrl->refocus(); });
	connect(ui->widgetControl, &WidgetControl::requestSR, this, [this] { ctrl->upsample(); });
	connect(ui->widgetControl, &WidgetControl::requestDE, this, [this] { ctrl->depth(); });
	connect(ui->widgetControl, &WidgetControl::requestChangingColor, this,
			[this](int index) { ctrl->colorChanged(index); });

	// 图像就绪
	connect(ctrl.get(), &LFControl::imageReady, ui->widgetImage, &WidgetImage::updateImage);

	// 启动业务
	// ctrl->params.path.white = "D:\\code\\LightFieldCamera\\B5152102610";
	// ctrl->readExtractLUT("data/calibration/lut_extract_9.bin");
	// ctrl->readDehexLUT("data/calibration/lut_dehex.bin");
	// ctrl->readLFP("data/toy.lfr");
	// ctrl->readWhite("data/MOD_0015.RAW");
	// ctrl->readSAI("data/bedroom");

	// normal
	// ctrl->readLFP("data/raw.png");
	// ctrl->readWhite("data/white.png");
	// ctrl->calibrate();
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::initStatusBar() {
	// 获取状态栏指针
	QStatusBar *bar = ui->statusbar;

	// ==========================================
	// 1. 创建控件 (Interactive Info)
	// ==========================================
	lblCoords = new QLabel("X: --  Y: --", this);
	lblRGB = new QLabel("R: --  G: --  B: --", this);
	lblZoom = new QLabel("100%", this);

	// ==========================================
	// 2. 创建控件 (Performance Metrics)
	// ==========================================
	lblResolution = new QLabel("RES: -- x --", this);
	lblFPS = new QLabel(QString("FPS: %1 / %2 / %3 (Cap/Proc/UI)").arg(0, 'f', 1).arg(0, 'f', 1).arg(0, 'f', 1), this);

	// ==========================================
	// 3. 美化样式
	// ==========================================
	// 设置最小宽度，避免数字跳动时界面抖动
	lblCoords->setMinimumWidth(100);
	lblRGB->setMinimumWidth(120);
	lblZoom->setMinimumWidth(50);
	lblFPS->setMinimumWidth(80);

	// 统一居中对齐
	// lblFPS->setAlignment(Qt::AlignCenter);

	// 设置边距和分割线视觉效果
	// 这里给每个 Label 加一点左右 padding，也可以加 border 来模拟分割线
	QString style = R"(
        QLabel { 
            padding-left: 5px; 
            padding-right: 5px; 
            color: #333333; 
        }
    )";

	lblCoords->setStyleSheet(style);
	lblRGB->setStyleSheet(style);
	lblZoom->setStyleSheet(style);
	lblResolution->setStyleSheet(style);
	lblFPS->setStyleSheet(style);

	// ==========================================
	// 4. 添加到状态栏 (顺序很重要！)
	// ==========================================
	// 左侧临时消息区是默认存在的，不需要add

	// 下面这些都会被加到【右侧区域】，且从左往右排

	// --- 靠左的部分 (交互信息) ---
	bar->addPermanentWidget(lblCoords);
	bar->addPermanentWidget(lblRGB);
	bar->addPermanentWidget(lblZoom);

	// --- 中间加个空的弹簧或者分割线 (可选) ---
	// bar->addPermanentWidget(new QLabel("|", this));

	// --- 最右的部分 (性能指标) ---
	bar->addPermanentWidget(lblResolution);
	bar->addPermanentWidget(lblFPS);

	// 初始化显示一条 Ready 消息
	bar->showMessage("Ready", 5000);
}

// ==========================================
// 槽函数实现：用于实时更新数据
// ==========================================

void MainWindow::updateMouseInfo(int x, int y, const QColor &color, double scale, int w, int h) {
	// --- 更新分辨率 (RES) ---
	// 如果没有图 (w=0)，显示 -- x --
	if (w > 0 && h > 0) {
		lblResolution->setText(QString("RES: %1 x %2").arg(w).arg(h));
	} else {
		lblResolution->setText("RES: -- x --");
	}

	// --- 更新缩放 (Zoom) ---
	int percent = static_cast<int>(scale * 100);
	lblZoom->setText(QString("%1%").arg(percent));

	// --- 更新坐标和颜色 (Coords & RGB) ---
	if (x < 0) {
		// 鼠标移出图片区域
		lblCoords->setText("X: --  Y: --");
		lblRGB->setText("R: --  G: --  B: --");
	} else {
		// 鼠标在图片上
		lblCoords->setText(QString("X: %1  Y: %2").arg(x).arg(y));
		lblRGB->setText(QString("R: %1  G: %2  B: %3").arg(color.red()).arg(color.green()).arg(color.blue()));
	}
	// m_uiFrameCount++;
}

void MainWindow::updateZoomInfo(double scale) {
	int percent = static_cast<int>(scale * 100);
	lblZoom->setText(QString("%1%").arg(percent));
}

void MainWindow::updateFPS(double fps) {
	lblFPS->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));

	// 简单的颜色警示：帧率太低变红
	if (fps < 15.0) {
		lblFPS->setStyleSheet("color: red; padding: 0 5px;");
	} else {
		lblFPS->setStyleSheet("color: green; padding: 0 5px;");
	}
}
