#ifndef LOGOUTPUT_H
#define LOGOUTPUT_H

#include <QWidget>

namespace Ui {
class WidgetLogger;
}

class WidgetLogger : public QWidget {
	Q_OBJECT

public:
	explicit WidgetLogger(QWidget *parent = nullptr);
	~WidgetLogger();

	void appendLog(int level, const QString &msg);

private slots:
	void save();

private:
	Ui::WidgetLogger *ui;

	QString getColorHtml(int level);
};

#endif // LOGOUTPUT_H
