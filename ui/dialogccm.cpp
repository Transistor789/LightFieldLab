#include "dialogccm.h"

#include "ui_dialogccm.h"

#include <QDoubleSpinBox>


DialogCCM::DialogCCM(std::vector<float> &data, QWidget *parent)
	: QDialog(parent), ui(new Ui::DialogCCM), ccmMatrix(data) {
	ui->setupUi(this);

	// 构造时立即加载数据到界面
	loadDataToUI();
}

DialogCCM::~DialogCCM() { delete ui; }

// 辅助函数：加载数据到 UI
void DialogCCM::loadDataToUI() {
	// 1. 安全检查
	if (ccmMatrix.size() < 9) {
		ccmMatrix.resize(9, 0.0f);
	}

	// 2. 使用数组简化 9 行赋值代码
	QDoubleSpinBox *boxes[] = {
		ui->doubleSpinBox00, ui->doubleSpinBox01, ui->doubleSpinBox02,
		ui->doubleSpinBox10, ui->doubleSpinBox11, ui->doubleSpinBox12,
		ui->doubleSpinBox20, ui->doubleSpinBox21, ui->doubleSpinBox22};

	for (int i = 0; i < 9; ++i) {
		if (boxes[i]) {
			boxes[i]->setValue(ccmMatrix[i]);
		}
	}
}

// 辅助函数：保存数据回 Vector
void DialogCCM::saveDataToVector() {
	// 同样使用数组简化保存逻辑
	QDoubleSpinBox *boxes[] = {
		ui->doubleSpinBox00, ui->doubleSpinBox01, ui->doubleSpinBox02,
		ui->doubleSpinBox10, ui->doubleSpinBox11, ui->doubleSpinBox12,
		ui->doubleSpinBox20, ui->doubleSpinBox21, ui->doubleSpinBox22};

	for (int i = 0; i < 9; ++i) {
		if (boxes[i]) {
			ccmMatrix[i] = static_cast<float>(boxes[i]->value());
		}
	}
}

void DialogCCM::accept() {
	// 点击 OK 时，将界面数据写回引用的 vector
	saveDataToVector();

	// 关闭窗口
	QDialog::accept();
}