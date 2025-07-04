#include "mainwindow.h"

#include "lfdata.h"
#include "lfload.h"
#include "lfprocessor.h"
#include "lfrefocus.h"
#include "lfsuperres.h"
#include "ui.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	std::cout << "MainWindow threadId: " << QThread::currentThreadId()
			  << std::endl;
	// ui
	ui = new Ui::MainWindow();
	ui->setupUi(this);
	// lfp
	lfp = new LFProcessor();
	lfp_thread = new QThread();
	lfp->moveToThread(lfp_thread);
	connect(lfp_thread, &QThread::started, lfp, &LFProcessor::printThreadId);
	connect(lfp_thread, &QThread::finished, lfp_thread, &QObject::deleteLater);
	connect(lfp_thread, &QThread::finished, lfp, &QObject::deleteLater);
	lfp_thread->start();

	connect(ui->gpuSlider, &QSlider::valueChanged, lfp,
			&LFProcessor::onGpuSliderValueChanged, Qt::QueuedConnection);

	// 0. load
	connect(ui->lensletBrowseBtn, &QPushButton::clicked, this,
			&MainWindow::onLensletBrowseBtn);
	onLensletBrowseBtn();
	connect(lfp->pLoad, &LFLoad::finished, this, &MainWindow::viewValueUpdated,
			Qt::QueuedConnection);
	// connect(lfp->load(), &LFLoad::finished, this,
	// &MainWindow::viewValueUpdated, 		Qt::QueuedConnection);

	// 1. view
	connect(ui->verticalSlider, &QSlider::valueChanged, this,
			&MainWindow::onViewVerticalSliderUpdated); // -> requestUpdateSAI
	connect(ui->horizontalSlider, &QSlider::valueChanged, this,
			&MainWindow::onViewHorizontalSliderUpdated); // -> requestUpdateSAI
	connect(this, &MainWindow::requestUpdateSAI, this, &MainWindow::updateSAI);
	connect(lfp, &LFProcessor::updateSAI, this, &MainWindow::updateSAI,
			Qt::QueuedConnection);

	// 2. refocus

	connect(ui->alphaSlider, &QSlider::valueChanged, this,
			&MainWindow::onRefocusAlphaChanged, Qt::QueuedConnection);
	connect(ui->cropSlider, &QSlider::valueChanged, this,
			&MainWindow::onRefocusCropChanged, Qt::QueuedConnection);
	// connect(lfp->getWorker<LFRefocus>(LFProcessor::REFOCUSER),
	// 		&LFRefocus::finished, this, &MainWindow::updateSAI,
	// 		Qt::QueuedConnection);
	connect(lfp->pRefocus, &LFRefocus::finished, this, &MainWindow::updateSAI,
			Qt::QueuedConnection);
	connect(lfp->pRefocus, &LFRefocus::finished, this, &MainWindow::updateSAI,
			Qt::QueuedConnection);

	// super resolution
	connect(ui->typeComboBox, &QComboBox::currentIndexChanged, lfp->pSuperres,
			&LFSuperres::setType, Qt::QueuedConnection);
	connect(ui->scaleComboBox, &QComboBox::currentIndexChanged, lfp->pSuperres,
			&LFSuperres::setScale, Qt::QueuedConnection);
	connect(ui->SRButton, &QPushButton::clicked, lfp,
			&LFProcessor::onSRButtonClicked, Qt::QueuedConnection);
	connect(lfp->pSuperres, &LFSuperres::finished, this, &MainWindow::updateSAI,
			Qt::QueuedConnection);
	// connect(lfp->pSuperres, &LFSuperres::finished, this,
	// 		[this](const cv::Mat &image_float) {
	// 			cv::Mat image_uint8;
	// 			image_float.convertTo(image_uint8,
	// 								  CV_8UC(lfp->lf_float->channels));
	// 			cv::imshow("image", image_uint8);
	// 			cv::waitKey(200);
	// 		});
}

MainWindow::~MainWindow() {
	if (lfp_thread->isRunning()) {
		lfp_thread->quit();
		lfp_thread->wait(); // 仅在必要时阻塞
	}
	delete ui;
}
void MainWindow::onLensletBrowseBtn() {
	if (ui->lensletPathEdit->text().isEmpty()) {
		return;
	}
	lfp->lensletImagePath = this->ui->lensletPathEdit->text();
	lfp->isRgb = this->ui->colorSlider->value() ? true : false;
	qDebug() << lfp->lensletImagePath;
	QMetaObject::invokeMethod(lfp->pLoad, "load", Qt::QueuedConnection,
							  Q_ARG(QString, lfp->lensletImagePath),
							  Q_ARG(bool, lfp->isRgb));
	// QMetaObject::invokeMethod(lfp->pLoad, &LFLoad::load,
	// Qt::QueuedConnection,
	// 						  Q_ARG(QString, lfp->lensletImagePath),
	// 						  Q_ARG(bool, lfp->isRgb));
}
void MainWindow::updateSAI(const cv::Mat &cvImg) {
	// std::cout << "MainWindow threadId: " << QThread::currentThreadId()
	// 		  << std::endl;

	// cv::imshow("SAI", cvImg);
	// cv::waitKey();

	QImage qImg;
	if (cvImg.channels() == 1) {
		qImg = QImage(cvImg.data, cvImg.cols, cvImg.rows, cvImg.step,
					  QImage::Format_Grayscale8);
	} else {
		qImg = QImage(cvImg.data, cvImg.cols, cvImg.rows, cvImg.step,
					  QImage::Format_BGR888);
	}
	ui->rightPanel->setPixmap(QPixmap::fromImage(qImg));
}
void MainWindow::viewValueUpdated(const LightFieldPtr &ptr) {
	int rows = ptr->rows, cols = ptr->cols;
	ui->verticalSlider->setRange(1, rows);
	ui->verticalSpinBox->setRange(1, rows);
	ui->horizontalSlider->setRange(1, cols);
	ui->horizontalSpinBox->setRange(1, cols);
	ui->verticalSlider->setValue(lfp->sai_row);
	ui->verticalSpinBox->setValue(lfp->sai_row);
	ui->horizontalSlider->setValue(lfp->sai_col);
	ui->horizontalSpinBox->setValue(lfp->sai_col);
	ui->cropSlider->setRange(0, rows / 2);
	ui->cropSpinBox->setRange(0, rows / 2);
}
void MainWindow::onViewVerticalSliderUpdated(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->sai_row = value - 1;
	emit requestUpdateSAI(lfp->lf->getSAI(lfp->sai_row, lfp->sai_col));
}
void MainWindow::onViewHorizontalSliderUpdated(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->sai_col = value - 1;
	emit requestUpdateSAI(lfp->lf->getSAI(lfp->sai_row, lfp->sai_col));
}
void MainWindow::onRefocusAlphaChanged(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->alpha = value / 100.0f;
	QMetaObject::invokeMethod(lfp->pRefocus, "refocus", Qt::QueuedConnection,
							  Q_ARG(float, lfp->alpha), Q_ARG(int, lfp->crop));
}
void MainWindow::onRefocusCropChanged(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->crop = value;
	QMetaObject::invokeMethod(lfp->pRefocus, "refocus", Qt::QueuedConnection,
							  Q_ARG(float, lfp->alpha), Q_ARG(int, lfp->crop));
}
