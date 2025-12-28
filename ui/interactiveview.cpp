#include "interactiveview.h"
#include <QWheelEvent>
#include <QScrollBar>
#include <cmath>

InteractiveView::InteractiveView(QWidget *parent) : QGraphicsView(parent)
{
	// 基础设置：无滚动条、抗锯齿、鼠标追踪
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setRenderHint(QPainter::Antialiasing);
	setRenderHint(QPainter::SmoothPixmapTransform);
	setDragMode(QGraphicsView::ScrollHandDrag);
	setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	setResizeAnchor(QGraphicsView::AnchorUnderMouse);
	setMouseTracking(true); // 关键：开启鼠标追踪
}

void InteractiveView::setImageItem(QGraphicsPixmapItem *item) {
	m_imageItem = item;
}

void InteractiveView::wheelEvent(QWheelEvent *event) {
	int angle = event->angleDelta().y();
	if (angle == 0) return;

	double scaleFactor = 1.15;
	if (angle > 0) scale(scaleFactor, scaleFactor);
	else scale(1.0 / scaleFactor, 1.0 / scaleFactor);

	emit zoomChanged(transform().m11());
	event->accept(); // 拦截事件，防止滚动条滚动
}

void InteractiveView::mouseMoveEvent(QMouseEvent *event) {
	QGraphicsView::mouseMoveEvent(event);

	if (m_imageItem) {
		// 坐标映射：视图 -> 场景 -> 图片像素
		QPointF scenePos = mapToScene(event->pos());
		QPointF itemPos = m_imageItem->mapFromScene(scenePos);
		QPoint pos(std::floor(itemPos.x()), std::floor(itemPos.y()));

		QPixmap pix = m_imageItem->pixmap();
		// 判断是否在图片范围内
		if (pos.x() >= 0 && pos.x() < pix.width() &&
			pos.y() >= 0 && pos.y() < pix.height()) {
			QColor color = pix.toImage().pixelColor(pos);
			emit mouseMoved(pos, color);
		} else {
			emit mouseMoved(QPoint(-1, -1), Qt::black);
		}
	}
}

void InteractiveView::mousePressEvent(QMouseEvent *event) {
	if (event->button() == Qt::LeftButton) setDragMode(QGraphicsView::ScrollHandDrag);
	QGraphicsView::mousePressEvent(event);
}

void InteractiveView::mouseReleaseEvent(QMouseEvent *event) {
	QGraphicsView::mouseReleaseEvent(event);
}
