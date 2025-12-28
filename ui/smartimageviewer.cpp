#include "smartimageviewer.h"

#include "ui_smartimageviewer.h"

#include <QFileDialog>
#include <QGraphicsPixmapItem>
#include <QStyle> // 用于获取内置图标
#include <QTimer>
#include <qnamespace.h>

SmartImageViewer::SmartImageViewer(QWidget *parent)
	: QWidget(parent), ui(new Ui::SmartImageViewer) {
	ui->setupUi(this);

	// 1. 初始化场景
	m_scene = new QGraphicsScene(this);
	m_scene->setBackgroundBrush(QColor(50, 50, 50)); // 深灰色背景

	m_scene->setBackgroundBrush(Qt::NoBrush);
	// ui->viewMain->setStyleSheet("background: transparent; border: none;");
	// ui->viewMain->setFrameShape(QFrame::NoFrame); // 去掉凹陷的边框

	// ui->viewMain 已经被提升为 InteractiveView 类型了
	ui->viewMain->setScene(m_scene);

	// 2. 连接信号 (UI 按钮的信号通过 "on_..._clicked" 自动连接，这里只连 View
	// 的)
	connect(ui->viewMain, &InteractiveView::mouseMoved, this,
			&SmartImageViewer::onMouseMoved);
	connect(ui->viewMain, &InteractiveView::zoomChanged, this,
			&SmartImageViewer::onZoomChanged);

	// 3. (可选) 给按钮设置一些系统自带的图标，好看点
	// ui->btnSave->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
	// ui->btnFit->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
	// ui->btnOrig->setIcon(
	// 	style()->standardIcon(QStyle::SP_TitleBarNormalButton));
}

SmartImageViewer::~SmartImageViewer() { delete ui; }

void SmartImageViewer::setImage(const QImage &img) {
	if (img.isNull())
		return;
	m_currentImage = img;

	m_scene->clear();
	QPixmap pix = QPixmap::fromImage(img);
	QGraphicsPixmapItem *item = m_scene->addPixmap(pix);

	ui->viewMain->setImageItem(item);
	m_scene->setSceneRect(pix.rect());

	// 【修改核心】：不要直接调用，而是延时 0 毫秒调用
	// 这会让 fitInView 在 UI 布局稳定后执行
	QTimer::singleShot(0, this, [this]() { on_btnFit_clicked(); });
}

void SmartImageViewer::setImage(const QString &path) { setImage(QImage(path)); }

// === 槽函数实现 ===

void SmartImageViewer::on_btnSave_clicked() {
	if (m_currentImage.isNull())
		return;
	QString path =
		QFileDialog::getSaveFileName(this, "保存", "", "Images (*.png *.jpg)");
	if (!path.isEmpty())
		m_currentImage.save(path);
}

void SmartImageViewer::on_btnFit_clicked() {
	if (m_scene->items().isEmpty())
		return;
	ui->viewMain->fitInView(m_scene->itemsBoundingRect(), Qt::KeepAspectRatio);
	// 更新缩放 Label
	onZoomChanged(ui->viewMain->transform().m11());
}

void SmartImageViewer::on_btnOrig_clicked() {
	ui->viewMain->resetTransform();
	ui->viewMain->scale(1.0, 1.0);
	onZoomChanged(1.0);
}

void SmartImageViewer::onMouseMoved(QPoint pos, QColor color) {
	if (pos.x() < 0) {
		ui->lblCoords->setText("X: -- Y: --");
		ui->lblPixel->setText("RGB: [--, --, --]");
	} else {
		ui->lblCoords->setText(
			QString("X: %1 Y: %2").arg(pos.x()).arg(pos.y()));
		ui->lblPixel->setText(QString("R:%1 G:%2 B:%3")
								  .arg(color.red())
								  .arg(color.green())
								  .arg(color.blue()));
	}
}

void SmartImageViewer::onZoomChanged(double scale) {
	ui->lblZoom->setText(QString::number(int(scale * 100)) + "%");
}

void SmartImageViewer::showEvent(QShowEvent *event) {
	QWidget::showEvent(event);

	// 稍微延时一点点，确保布局（Geometry）已经完全计算完毕
	// 解决“缩略图”问题的关键
	if (!m_scene->items().isEmpty()) {
		on_btnFit_clicked();
	}
}

// 【修复核心 2】当用户拖拽改变窗口大小时，图像自动跟随适应
void SmartImageViewer::resizeEvent(QResizeEvent *event) {
	QWidget::resizeEvent(event);

	// 如果你希望窗口变大时，图片也自动变大填满，就加上这一句
	if (!m_scene->items().isEmpty()) {
		on_btnFit_clicked();
	}
}