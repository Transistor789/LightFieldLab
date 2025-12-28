#ifndef SMARTIMAGEVIEWER_H
#define SMARTIMAGEVIEWER_H

#include <QGraphicsScene>
#include <QImage>
#include <QWidget>

namespace Ui {
class SmartImageViewer;
}

class SmartImageViewer : public QWidget {
	Q_OBJECT

public:
	explicit SmartImageViewer(QWidget *parent = nullptr);
	~SmartImageViewer();

	// 公开接口
	void setImage(const QImage &img);
	void setImage(const QString &path);

protected:
	void showEvent(QShowEvent *event) override;		// 窗口显示时触发
	void resizeEvent(QResizeEvent *event) override; // 窗口大小改变时触发

private slots:
	// UI 按钮的槽函数
	void on_btnSave_clicked();
	void on_btnFit_clicked();
	void on_btnOrig_clicked();

	// 接收内部 View 的信号
	void onMouseMoved(QPoint pos, QColor color);
	void onZoomChanged(double scale);

private:
	Ui::SmartImageViewer *ui;
	QGraphicsScene *m_scene;
	QImage m_currentImage;
};

#endif // SMARTIMAGEVIEWER_H