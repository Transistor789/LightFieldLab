#include "dialogwbgains.h"

#include "ui_dialogwbgains.h"

#include <QDoubleSpinBox>

DialogWBGains::DialogWBGains(std::vector<float> &data, QWidget *parent)
	: QDialog(parent), ui(new Ui::DialogWBGains), wbGains(data) {
	ui->setupUi(this);

	// 构造时立即加载数据
	loadDataToUI();
}

DialogWBGains::~DialogWBGains() { delete ui; }

void DialogWBGains::loadDataToUI() {
	// 1. 确保 vector 大小足够 (通常 RGGB 需要 4 个值)
	if (wbGains.size() < 4) {
		wbGains.resize(4, 1.0f); // 默认为 1.0
	}

	// 2. 将 UI 控件放入数组方便循环
	QDoubleSpinBox *boxes[] = {ui->gain1, ui->gain2, ui->gain3, ui->gain4};

	// 3. 赋值
	for (int i = 0; i < 4; ++i) {
		if (boxes[i]) {
			boxes[i]->setValue(wbGains[i]);
		}
	}
}

void DialogWBGains::accept() {
	// 点击 OK 时，将 UI 数据保存回 wbGains 引用
	QDoubleSpinBox *boxes[] = {ui->gain1, ui->gain2, ui->gain3, ui->gain4};

	for (int i = 0; i < 4; ++i) {
		if (boxes[i]) {
			wbGains[i] = static_cast<float>(boxes[i]->value());
		}
	}

	// 关闭窗口
	QDialog::accept();
}