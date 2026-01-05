#ifndef SMARTIMAGEVIEWER_H
#define SMARTIMAGEVIEWER_H

#include <QGraphicsScene>
#include <QImage>
#include <QMenu>
#include <QWidget>

namespace Ui {
class SmartImageViewer;
}

class SmartImageViewer : public QWidget {
	Q_OBJECT

public:
	explicit SmartImageViewer(QWidget *parent = nullptr);
	~SmartImageViewer();

	void setImage(const QImage &img);
	void setImage(const QString &path);

signals:
	void mouseMoved(int x, int y, QColor color, double scale, int w, int h);
	void zoomChanged(double scale);

protected:
	void showEvent(QShowEvent *event) override;
	void resizeEvent(QResizeEvent *event) override;

private slots:
	void onShowContextMenu(const QPoint &pos);
	void onMouseMoved(QPoint pos, QColor color);
	void onZoomChanged(double scale);

private:
	void on_btnSave_clicked();
	void on_btnFit_clicked();
	void on_btnOrig_clicked();

private:
	Ui::SmartImageViewer *ui;
	QGraphicsScene *m_scene;
	QImage m_currentImage;
};

#endif // SMARTIMAGEVIEWER_H