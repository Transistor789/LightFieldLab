#ifndef WIDGETIMAGE_H
#define WIDGETIMAGE_H

#include "smartimageviewer.h"

#include <QMap>
#include <QWidget>

enum class ImageType { LFP, White, Center, Refocus, SR, Depth };

namespace Ui {
class WidgetImage;
}

class WidgetImage : public QWidget {
	Q_OBJECT

public:
	explicit WidgetImage(QWidget *parent = nullptr);
	~WidgetImage();

	void updateImage(ImageType type, const QImage &img);

private:
	Ui::WidgetImage *ui;

	QMap<ImageType, SmartImageViewer *> m_viewers;
};

#endif // IMAGEWORKSPACE_H
