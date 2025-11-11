#ifndef UI_H
#define UI_H

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QSpinBox>

namespace Ui {
class MainWindow : public QMainWindow {
	Q_OBJECT
public:
	void setupUi(QMainWindow *mainWindow);

	// 1
	QSlider *captureSlider, *colorSlider, *gpuSlider;

	QRadioButton *staticMode;
	QRadioButton *dynamicMode;
	QRadioButton *grayMode;
	QRadioButton *rgbMode;
	QPushButton *lensletBrowseBtn;
	QPushButton *whiteBrowseBtn;
	QLineEdit *lensletPathEdit;
	QLineEdit *whitePathEdit;

	// views
	QSlider *verticalSlider;
	QSlider *horizontalSlider;
	QSpinBox *verticalSpinBox;
	QSpinBox *horizontalSpinBox;

	// refocus
	QSlider *cropSlider;
	QSpinBox *cropSpinBox;
	QSlider *alphaSlider;
	QDoubleSpinBox *alphaSpinBox;

	// super_resolution
	QComboBox *typeComboBox;
	QComboBox *scaleComboBox;
	QPushButton *SRButton;

	QLabel *rightPanel;

private:
	QGroupBox *setupModeGroup();
	QGroupBox *setupViewsGroup();
	QGroupBox *setupRefocusGroup();
	QGroupBox *setupSRGroup();
	QGroupBox *setupDEGroup();
};
}; // namespace Ui

#endif // UI_H
