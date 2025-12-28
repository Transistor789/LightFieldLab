#ifndef INTERACTIVEVIEW_H
#define INTERACTIVEVIEW_H

#include <QGraphicsView>
#include <QGraphicsPixmapItem>

class InteractiveView : public QGraphicsView
{
	Q_OBJECT
public:
	explicit InteractiveView(QWidget *parent = nullptr);
	void setImageItem(QGraphicsPixmapItem *item); // 设置要追踪的 Item

signals:
	// 发送信号给 UI 更新坐标和颜色
	void mouseMoved(QPoint pos, QColor color);
	void zoomChanged(double scale);

protected:
	void wheelEvent(QWheelEvent *event) override;        // 滚轮缩放
	void mouseMoveEvent(QMouseEvent *event) override;    // 移动追踪
	void mousePressEvent(QMouseEvent *event) override;   // 拖拽
	void mouseReleaseEvent(QMouseEvent *event) override; // 拖拽

private:
	QGraphicsPixmapItem *m_imageItem = nullptr;
};

#endif // INTERACTIVEVIEW_H
