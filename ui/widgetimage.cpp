#include "widgetimage.h"

#include "ui_widgetimage.h"

#include <QWheelEvent>

WidgetImage::WidgetImage(QWidget *parent)
	: QWidget(parent), ui(new Ui::WidgetImage) {
	ui->setupUi(this);

	m_viewers = {{ImageType::LFP, ui->widgetLFP},
				 {ImageType::White, ui->widgetWhite},
				 {ImageType::Center, ui->widgetCenter},
				 {ImageType::Refocus, ui->widgetRefocus},
				 {ImageType::SR, ui->widgetSR},
				 {ImageType::Depth, ui->widgetDepth}};

	QList<SmartImageViewer *> allViewers = findChildren<SmartImageViewer *>();
	for (SmartImageViewer *viewer : allViewers) {
		connect(viewer, &SmartImageViewer::mouseMoved, this,
				&WidgetImage::imageMouseMoved);
		connect(viewer, &SmartImageViewer::zoomChanged, this,
				&WidgetImage::imageZoomChanged);
	}
}

WidgetImage::~WidgetImage() { delete ui; }

void WidgetImage::updateImage(ImageType type, const QImage &img) {
	if (m_viewers.contains(type)) {
		m_viewers[type]->setImage(img);
	}
}

// void WidgetImage::showLFP(const QImage &img) { ui->widgetLFP->setImage(img);
// }

// void WidgetImage::showWhite(const QImage &img) {
// 	ui->widgetWhite->setImage(img);
// }

// void WidgetImage::showCenter(const QImage &img) {
// 	ui->widgetCenter->setImage(img);
// }

// void WidgetImage::showRefocus(const QImage &img) {
// 	ui->widgetRefocus->setImage(img);
// }

// void WidgetImage::showSR(const QImage &img) { ui->widgetSR->setImage(img); }

// void WidgetImage::showDepth(const QImage &img) {
// 	ui->widgetDepth->setImage(img);
// }
