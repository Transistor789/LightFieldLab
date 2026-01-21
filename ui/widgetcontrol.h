#ifndef WIDGETCONTROL_H
#define WIDGETCONTROL_H

#include "lfparams.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QRadioButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QTextEdit>
#include <QWidget>
#include <qcontainerfwd.h>
#include <qtmetamacros.h>
#include <string>
#include <type_traits>

namespace Ui {
class WidgetControl;
}

class WidgetControl : public QWidget {
	Q_OBJECT

public:
	explicit WidgetControl(QWidget *parent = nullptr);
	~WidgetControl();

	void setupParams(LFParams *params);
public slots:
	void updateUI();

signals:
	// 【关键】定义一个信号，把路径传出去
	void requestLoadLFP(const QString &path);
	void requestLoadWhite(const QString &path);
	void requestLoadSAI(const QString &path);
	void requestLoadExtractLUT(const QString &path);
	void requestLoadDehexLUT(const QString &path);

	void requestCalibrate();
	void requestFastPreview();
	void requestISP();
	void requestDetectCamera();
	void requestCapture(bool active);
	void requestProcess(bool active);
	void requestSaveSAI(const QString &path);
	void requestPlay();
	void requestSAI(int row, int col);
	void requestRefocus();
	void requestSR();
	void requestSRModel(const QString &path);
	void requestDE();
	void requestDEModel(const QString &path);
	void requestChangingColor(int index);

	void setRefocusCrop(int value);
	void setRefocusAlpha(double value);

private:
	Ui::WidgetControl *ui;

	LFParams *params_ = nullptr;
};

template <typename TWidget, typename TVal>
void setValSilent(TWidget *w, TVal val) {
	if (!w)
		return;
	QSignalBlocker blocker(w); // 自动屏蔽信号

	// 1. 数值类控件 (setValue)
	// 包含: SpinBox, Slider, ProgressBar, Dial
	if constexpr (std::is_same_v<TWidget, QSpinBox> || std::is_same_v<TWidget, QDoubleSpinBox>
				  || std::is_same_v<TWidget, QSlider> || std::is_same_v<TWidget, QProgressBar>) {
		w->setValue(val);
	}

	// 2. 开关/状态类控件 (setChecked)
	// 包含: CheckBox, RadioButton, 可勾选的 GroupBox,
	// Action(虽不是Widget但常用)
	else if constexpr (std::is_same_v<TWidget, QCheckBox> || std::is_same_v<TWidget, QRadioButton>
					   || std::is_same_v<TWidget, QGroupBox>) {
		w->setChecked(val);
	}

	// 3. 索引类控件 (setCurrentIndex)
	// 包含: ComboBox, TabWidget, StackedWidget
	else if constexpr (std::is_same_v<TWidget, QComboBox> || std::is_same_v<TWidget, QTabWidget>
					   || std::is_same_v<TWidget, QStackedWidget>) {
		w->setCurrentIndex(static_cast<int>(val));
	}

	// 4. 单行文本类 (setText)
	// 包含: Label, LineEdit
	else if constexpr (std::is_same_v<TWidget, QLabel> || std::is_same_v<TWidget, QLineEdit>) {
		// 智能适配：如果是数字，自动转 QString；如果是字符串，直接设置
		if constexpr (std::is_arithmetic_v<TVal>) {
			w->setText(QString::number(val));
		} else if constexpr (std::is_same_v<TVal, std::string>) {
			w->setText(QString::fromStdString(val));
		} else {
			w->setText(val);
		}
	}

	// 5. 多行文本类 (setPlainText / setText)
	// 包含: TextEdit, PlainTextEdit
	else if constexpr (std::is_same_v<TWidget, QPlainTextEdit>) {
		if constexpr (std::is_arithmetic_v<TVal>) {
			w->setPlainText(QString::number(val));
		} else {
			w->setPlainText(val);
		}
	} else if constexpr (std::is_same_v<TWidget, QTextEdit>) {
		if constexpr (std::is_arithmetic_v<TVal>) {
			w->setText(QString::number(val));
		} else {
			w->setText(val);
		}
	}
}

#endif // WIDGETCONTROL_H
