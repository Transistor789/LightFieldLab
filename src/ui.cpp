#include "ui.h"

namespace Ui {
void MainWindow::setupUi(QMainWindow *mainWindow) {
	QWidget *centralWidget = new QWidget(mainWindow);
	mainWindow->setCentralWidget(centralWidget);
	centralWidget->setMinimumSize(1280, 720);
	QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

	// === 左侧功能区 ===
	QWidget *leftPanel = new QWidget();
	QVBoxLayout *leftLayout = new QVBoxLayout(leftPanel);
	leftPanel->setMinimumWidth(300); // 关键修改点2：设置最小宽度

	QScrollArea *leftScrollArea = new QScrollArea();
	leftScrollArea->setWidgetResizable(true); // 重要：允许内部widget调整大小
	leftScrollArea->setHorizontalScrollBarPolicy(
		Qt::ScrollBarAlwaysOff); // 禁用水平滚动条
	leftScrollArea->setVerticalScrollBarPolicy(
		Qt::ScrollBarAsNeeded); // 启用垂直滚动条

	leftLayout->addWidget(setupModeGroup());
	leftLayout->addWidget(setupViewsGroup());
	leftLayout->addWidget(setupRefocusGroup());
	leftLayout->addWidget(setupSRGroup());
	leftLayout->addWidget(setupDEGroup());

	leftScrollArea->setWidget(leftPanel); // 将容器放入滚动区域

	// === 右侧显示区 ===
	rightPanel = new QLabel("图像显示区域");
	rightPanel->setAlignment(Qt::AlignCenter);
	rightPanel->setStyleSheet("border: 1px solid gray;");

	// 主布局组装
	mainLayout->addWidget(leftScrollArea, 1);
	mainLayout->addWidget(rightPanel, 3);
}
// 1. 数据模式组
QGroupBox *MainWindow::setupModeGroup() {
	QGroupBox *group = new QGroupBox("数据模式");
	QVBoxLayout *layout = new QVBoxLayout();

	// 1. static/dynamic
	QVBoxLayout *captureLayout = new QVBoxLayout();
	QLabel *staticLabel = new QLabel("静态");
	QLabel *dynamicLabel = new QLabel("动态");
	QHBoxLayout *captureLabelLayout = new QHBoxLayout();
	captureLabelLayout->addWidget(staticLabel);
	captureLabelLayout->addWidget(dynamicLabel);
	captureSlider = new QSlider(Qt::Horizontal);
	captureSlider->setRange(0, 1);
	captureSlider->setValue(0);
	captureLayout->addLayout(captureLabelLayout);
	captureLayout->addWidget(captureSlider);

	// 2. gray/rgb
	QVBoxLayout *colorLayout = new QVBoxLayout();
	QLabel *grayLabel = new QLabel("gray");
	QLabel *rgbLabel = new QLabel("rgb");
	QHBoxLayout *colorLabelLayout = new QHBoxLayout();
	colorLabelLayout->addWidget(grayLabel);
	colorLabelLayout->addWidget(rgbLabel);
	colorSlider = new QSlider(Qt::Horizontal);
	colorSlider->setRange(0, 1);
	colorSlider->setValue(1);
	colorLayout->addLayout(colorLabelLayout);
	colorLayout->addWidget(colorSlider);

	// 3. cpu/gpu
	QVBoxLayout *gpuLayout = new QVBoxLayout();
	QLabel *cpuLabel = new QLabel("cpu");
	QLabel *gpuLabel = new QLabel("gpu");
	QHBoxLayout *gpuLabelLayout = new QHBoxLayout();
	gpuLabelLayout->addWidget(cpuLabel);
	gpuLabelLayout->addWidget(gpuLabel);
	gpuSlider = new QSlider(Qt::Horizontal);
	gpuSlider->setRange(0, 1);
	gpuSlider->setValue(0);
	gpuLayout->addLayout(gpuLabelLayout);
	gpuLayout->addWidget(gpuSlider);

	// 组装布局
	QHBoxLayout *optionLayout = new QHBoxLayout();
	optionLayout->addLayout(captureLayout);
	optionLayout->addLayout(colorLayout);
	optionLayout->addLayout(gpuLayout);
	layout->addLayout(optionLayout);

	group->setLayout(layout);

	// 输入区域堆叠布局
	QStackedWidget *inputStack = new QStackedWidget(group);

	// --- 静态模式页面 ---
	QWidget *staticPage = new QWidget();
	QVBoxLayout *staticLayout = new QVBoxLayout(staticPage);

	QHBoxLayout *whiteLayout = new QHBoxLayout();
	whitePathEdit = new QLineEdit();
	whitePathEdit->setPlaceholderText("白图像路径");
	whiteBrowseBtn = new QPushButton("...");
	whiteBrowseBtn->setFixedWidth(30); // 固定按钮宽度
	whiteLayout->addWidget(whitePathEdit);
	whiteLayout->addWidget(whiteBrowseBtn);

	QHBoxLayout *lensletLayout = new QHBoxLayout();
	lensletPathEdit = new QLineEdit("input/toy");
	lensletPathEdit->setPlaceholderText("微透镜图像路径");
	lensletBrowseBtn = new QPushButton("...");
	lensletBrowseBtn->setFixedWidth(30);
	lensletLayout->addWidget(lensletPathEdit);
	lensletLayout->addWidget(lensletBrowseBtn);

	staticLayout->addLayout(lensletLayout);
	staticLayout->addLayout(whiteLayout);
	inputStack->addWidget(staticPage);

	// --- 动态模式页面 ---
	QWidget *dynamicPage = new QWidget();
	QHBoxLayout *dynamicLayout = new QHBoxLayout(dynamicPage);
	QPushButton *btnCapture = new QPushButton("采集");
	QPushButton *btnDecode = new QPushButton("解码");
	dynamicLayout->addWidget(btnCapture);
	dynamicLayout->addWidget(btnDecode);
	inputStack->addWidget(dynamicPage);

	layout->addWidget(inputStack);

	connect(captureSlider, &QSlider::valueChanged, this,
			[inputStack](int isStatic) {
				inputStack->setCurrentIndex(isStatic ? 1 : 0);
			});
	connect(whiteBrowseBtn, &QPushButton::clicked,
			[this, whitePathEdit = this->whitePathEdit]() {
				QString path = QFileDialog::getOpenFileName(
					this, "选择白图像", "",
					"Images (*jpg *jpeg *.png *.tiff *.bmp *.mim)");
				if (!path.isEmpty()) {
					whitePathEdit->setText(path);
				}
			});
	connect(
		lensletBrowseBtn, &QPushButton::clicked,
		[this, lensletPathEdit = this->lensletPathEdit]() {
			QString path = QFileDialog::getExistingDirectory(
				this, "选择微透镜图像所在文件夹", "",
				QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
			if (!path.isEmpty()) {
				lensletPathEdit->setText(path);
			}
		});
	return group;
}

QGroupBox *MainWindow::setupViewsGroup() {
	QGroupBox *group = new QGroupBox("视角变换");
	QVBoxLayout *layout = new QVBoxLayout();

	QLabel *verticalLabel = new QLabel("垂直");
	QLabel *horizontalLabel = new QLabel("水平");
	QHBoxLayout *verticalLayout = new QHBoxLayout();
	verticalSlider = new QSlider(Qt::Horizontal);
	verticalSlider->setRange(1, 15);
	verticalSlider->setValue(8);
	verticalSpinBox = new QSpinBox();
	verticalSpinBox->setRange(1, 15);
	verticalSpinBox->setValue(8);
	verticalSpinBox->setSingleStep(1);
	verticalLayout->addWidget(verticalSlider);
	verticalLayout->addWidget(verticalSpinBox);

	QHBoxLayout *horizontalLayout = new QHBoxLayout();
	horizontalSlider = new QSlider(Qt::Horizontal);
	horizontalSlider->setRange(1, 15);
	horizontalSlider->setValue(8);
	horizontalSpinBox = new QSpinBox();
	horizontalSpinBox->setRange(1, 15);
	horizontalSpinBox->setValue(8);
	horizontalSpinBox->setSingleStep(1);
	horizontalLayout->addWidget(horizontalSlider);
	horizontalLayout->addWidget(horizontalSpinBox);

	layout->addWidget(verticalLabel);
	layout->addLayout(verticalLayout);
	layout->addWidget(horizontalLabel);
	layout->addLayout(horizontalLayout);
	group->setLayout(layout);
	connect(verticalSlider, &QSlider::valueChanged, verticalSpinBox,
			[=](int value) { verticalSpinBox->setValue(value); });
	connect(verticalSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			verticalSlider,
			[=](int value) { verticalSlider->setValue(value); });
	connect(horizontalSlider, &QSlider::valueChanged, horizontalSpinBox,
			[=](int value) { horizontalSpinBox->setValue(value); });
	connect(horizontalSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			horizontalSlider,
			[=](int value) { horizontalSlider->setValue(value); });

	return group;
}

// 3. 重聚焦组
QGroupBox *MainWindow::setupRefocusGroup() {
	QGroupBox *group = new QGroupBox("重聚焦");
	QVBoxLayout *layout = new QVBoxLayout();

	QHBoxLayout *cropLayout = new QHBoxLayout();
	QLabel *cropLabel = new QLabel("crop");
	QLabel *alphaLabel = new QLabel("alpha");
	cropSlider = new QSlider(Qt::Horizontal);
	cropSlider->setRange(0, 7);
	cropSlider->setValue(0);
	cropSpinBox = new QSpinBox();
	cropSpinBox->setRange(0, 7);
	cropSpinBox->setValue(0);
	cropSpinBox->setSingleStep(1);
	cropLayout->addWidget(cropSlider);
	cropLayout->addWidget(cropSpinBox);

	QHBoxLayout *alphaLayout = new QHBoxLayout();
	alphaSlider = new QSlider(Qt::Horizontal);
	alphaSlider->setRange(0, 300); // 0.00~3.00 映射为 0~300
	alphaSlider->setValue(100);	   // 默认值 1.50
	alphaSpinBox = new QDoubleSpinBox();
	alphaSpinBox->setRange(0.0, 3.0);  // 范围
	alphaSpinBox->setDecimals(2);	   // 保留x位小数
	alphaSpinBox->setValue(1.0);	   // 默认值
	alphaSpinBox->setSingleStep(0.01); // 步长
	alphaLayout->addWidget(alphaSlider);
	alphaLayout->addWidget(alphaSpinBox);

	layout->addWidget(cropLabel);
	layout->addLayout(cropLayout);
	layout->addWidget(alphaLabel);
	layout->addLayout(alphaLayout);
	group->setLayout(layout);

	connect(cropSlider, &QSlider::valueChanged, cropSpinBox,
			[=](int value) { cropSpinBox->setValue(value); });
	connect(cropSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			cropSlider, [=](int value) { cropSlider->setValue(value); });
	connect(alphaSlider, &QSlider::valueChanged, alphaSpinBox, [=](int value) {
		alphaSpinBox->setValue(value / 100.0); // 将0~300映射为0.00~3.00
	});
	connect(alphaSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			alphaSlider, [=](double value) {
				alphaSlider->setValue(
					static_cast<int>(value * 100)); // 将0.00~3.00映射为0~300
			});
	return group;
}

// 4. 超分辨组
QGroupBox *MainWindow::setupSRGroup() {
	QGroupBox *group = new QGroupBox("超分辨");
	QHBoxLayout *layout = new QHBoxLayout();
	group->setLayout(layout);

	typeComboBox = new QComboBox(group);
	typeComboBox->addItem("Nearest", 0);
	typeComboBox->addItem("Linear", 1);
	typeComboBox->addItem("Cubic", 2);
	typeComboBox->addItem("Lanczos", 3);
	typeComboBox->addItem("EDSR", 4);
	typeComboBox->addItem("ESPCN", 5);
	typeComboBox->addItem("FSRCNN", 6);
	layout->addWidget(typeComboBox);

	scaleComboBox = new QComboBox(group);
	scaleComboBox->addItem("2x", 2);
	scaleComboBox->addItem("3x", 3);
	scaleComboBox->addItem("4x", 4);
	layout->addWidget(scaleComboBox);

	SRButton = new QPushButton("设置", group);
	layout->addWidget(SRButton);

	return group;
}

// 5. 深度估计组
QGroupBox *MainWindow::setupDEGroup() {
	QGroupBox *group = new QGroupBox("深度估计");
	QVBoxLayout *layout = new QVBoxLayout();
	group->setLayout(layout);

	return group;
}

}; // namespace Ui
