#ifndef WIDGETCONTROL_H
#define WIDGETCONTROL_H

#include <QWidget>

namespace Ui {
class WidgetControl;
}

class WidgetControl : public QWidget {
	Q_OBJECT

public:
	explicit WidgetControl(QWidget *parent = nullptr);
	~WidgetControl();

signals:
	// 【关键】定义一个信号，把路径传出去
	void requestLoadLFP(const QString &path);
	void requestLoadWhite(const QString &path);
	void requestLoadSAI(const QString &path);
	void requestLoadSliceLUT(const QString &path);
	void requestLoadDehexLUT(const QString &path);

private:
	Ui::WidgetControl *ui;
};

#endif // WIDGETCONTROL_H
