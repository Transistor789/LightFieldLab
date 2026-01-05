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

	// ... (原有初始化场景代码保持不变) ...
	m_scene = new QGraphicsScene(this);
	m_scene->setBackgroundBrush(QColor(50, 50, 50));
	m_scene->setBackgroundBrush(Qt::NoBrush);
	ui->viewMain->setScene(m_scene);

	// ... (原有信号连接保持不变) ...
	connect(ui->viewMain, &InteractiveView::mouseMoved, this,
			&SmartImageViewer::onMouseMoved);
	connect(ui->viewMain, &InteractiveView::zoomChanged, this,
			&SmartImageViewer::onZoomChanged);

	// ================== [新增部分] ==================

	// 1. 设置 View 的菜单策略为自定义
	ui->viewMain->setContextMenuPolicy(Qt::CustomContextMenu);

	// 2. 连接右键请求信号到槽函数
	connect(ui->viewMain, &InteractiveView::customContextMenuRequested, this,
			&SmartImageViewer::onShowContextMenu);

	// ===============================================
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
	// 1. 获取缩放比例
	// viewMain 的 transform().m11() 就是水平缩放系数
	double scale = ui->viewMain->transform().m11();

	// 2. 获取图像分辨率
	int w = 0, h = 0;
	if (!m_currentImage.isNull()) {
		w = m_currentImage.width();
		h = m_currentImage.height();
	}

	// 3. [修改] 打包发送所有数据
	emit mouseMoved(pos.x(), pos.y(), color, scale, w, h);
}

void SmartImageViewer::onZoomChanged(double scale) {
	// 1. (可选) 更新自己的 UI
	// ui->lblZoom->setText(QString::number(int(scale * 100)) + "%");

	// 2. [新增] 转发信号
	emit zoomChanged(scale);
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

void SmartImageViewer::onShowContextMenu(const QPoint &pos) {
	QMenu contextMenu(this);

	// 添加动作
	QAction *actionFit =
		contextMenu.addAction(QIcon::fromTheme("zoom-fit-best"), "适应窗口");
	QAction *actionOrig = contextMenu.addAction(
		QIcon::fromTheme("zoom-original"), "原始大小 (1:1)");
	contextMenu.addSeparator(); // 分隔线
	QAction *actionSave =
		contextMenu.addAction(QIcon::fromTheme("document-save"), "另存为...");

	// 连接动作到已有的槽函数 (复用逻辑)
	connect(actionFit, &QAction::triggered, this,
			&SmartImageViewer::on_btnFit_clicked);
	connect(actionOrig, &QAction::triggered, this,
			&SmartImageViewer::on_btnOrig_clicked);
	connect(actionSave, &QAction::triggered, this,
			&SmartImageViewer::on_btnSave_clicked);

	// 状态检查：如果没有图片，禁用某些选项
	if (m_scene->items().isEmpty()) {
		actionFit->setEnabled(false);
		actionOrig->setEnabled(false);
		actionSave->setEnabled(false);
	}

	// 在鼠标点击位置显示菜单
	// 注意：pos 是 viewMain 的局部坐标，需要映射到全局屏幕坐标
	contextMenu.exec(ui->viewMain->mapToGlobal(pos));
}