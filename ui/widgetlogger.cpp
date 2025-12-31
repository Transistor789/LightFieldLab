#include "widgetlogger.h"

#include "ui_widgetlogger.h"

#include <QFile>
#include <QFileDialog>
#include <QScrollBar>
#include <qpushbutton.h>

WidgetLogger::WidgetLogger(QWidget *parent)
	: QWidget(parent), ui(new Ui::WidgetLogger) {
	ui->setupUi(this);

	// 设置最大行数，防止内存无限膨胀 (比如保留最近 5000 行)
	ui->textEdit->document()->setMaximumBlockCount(5000);

	connect(ui->btnClear, &QPushButton::clicked, this,
			[this] { ui->textEdit->clear(); });
	connect(ui->btnSave, &QPushButton::clicked, this, &WidgetLogger::save);

	connect(ui->btnHide, &QPushButton::toggled, this, [this](bool value) {
		ui->textEdit->setVisible(!value);
		ui->btnHide->setText(value ? "显示日志" : "隐藏日志");
	});
	ui->btnHide->click();
}

WidgetLogger::~WidgetLogger() { delete ui; }

void WidgetLogger::appendLog(int level, const QString &msg) {
	// 1. 获取颜色
	QString color = getColorHtml(level);

	// 2. 构造 HTML 字符串 (例如: <font color="red">Error: ...</font>)
	// 这里的 msg 已经是 GlobalLogger 发过来的带时间戳的格式化字符串
	QString html = QString("<font color=\"%1\">%2</font>").arg(color, msg);

	// 3. 追加到文本框
	ui->textEdit->append(html);

	// 4. 处理自动滚动
	// 如果勾选了 AutoScroll (假设 checkbox 叫 chkAutoScroll)
	if (ui->chkAutoScroll->isChecked()) {
		QScrollBar *sb = ui->textEdit->verticalScrollBar();
		sb->setValue(sb->maximum());
	}
}

QString WidgetLogger::getColorHtml(int level) {
	switch (level) {
		case 0:
			return "#000000"; // Info: 黑色 (深色模式下建议用白色或浅灰)
		case 1:
			return "#CC9900"; // Warn: 深黄色/橙色
		case 2:
			return "#FF0000"; // Error: 红色
		case 3:
			return "#0000FF"; // Debug: 蓝色
		default:
			return "black";
	}
}

void WidgetLogger::save() {
	QString filename = QFileDialog::getSaveFileName(this, "Save Log", "",
													"Text Files (*.txt)");
	if (filename.isEmpty())
		return;

	QFile file(filename);
	if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		QTextStream out(&file);
		out << ui->textEdit->toPlainText(); // 保存纯文本，不要 HTML 标签
		file.close();
	}
}
