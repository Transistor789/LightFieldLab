#include "dialogccm.h"

#include "ui_dialogccm.h"


DialogCCM::DialogCCM(QWidget *parent) : QDialog(parent), ui(new Ui::DialogCCM) {
	ui->setupUi(this);

	// 可以在这里设置 SpinBox 的范围、步长等通用属性
	// 例如：
	// QList<QDoubleSpinBox*> spins = this->findChildren<QDoubleSpinBox*>();
	// for(auto s : spins) { s->setRange(-10.0, 10.0); s->setSingleStep(0.1); }
}

DialogCCM::~DialogCCM() { delete ui; }

void DialogCCM::setupParams(LFParamsSource *params) {
	params_ = params;
	updateUI(); // 加载数据到界面
}

void DialogCCM::updateUI() {
	if (!params_)
		return;

	// 确保 vector 大小足够，防止越界
	if (params_->ccm_matrix.size() < 9) {
		params_->ccm_matrix.resize(9, 0.0f);
	}

	// 将 vector (行优先) 数据加载到 doubleSpinBoxXX
	// Index = row * 3 + col

	// Row 0
	ui->doubleSpinBox00->setValue(params_->ccm_matrix[0]);
	ui->doubleSpinBox01->setValue(params_->ccm_matrix[1]);
	ui->doubleSpinBox02->setValue(params_->ccm_matrix[2]);

	// Row 1
	ui->doubleSpinBox10->setValue(params_->ccm_matrix[3]);
	ui->doubleSpinBox11->setValue(params_->ccm_matrix[4]);
	ui->doubleSpinBox12->setValue(params_->ccm_matrix[5]);

	// Row 2
	ui->doubleSpinBox20->setValue(params_->ccm_matrix[6]);
	ui->doubleSpinBox21->setValue(params_->ccm_matrix[7]);
	ui->doubleSpinBox22->setValue(params_->ccm_matrix[8]);
}

void DialogCCM::accept() {
	if (params_) {
		// 将 UI 数据保存回 vector
		params_->ccm_matrix[0] =
			static_cast<float>(ui->doubleSpinBox00->value());
		params_->ccm_matrix[1] =
			static_cast<float>(ui->doubleSpinBox01->value());
		params_->ccm_matrix[2] =
			static_cast<float>(ui->doubleSpinBox02->value());

		params_->ccm_matrix[3] =
			static_cast<float>(ui->doubleSpinBox10->value());
		params_->ccm_matrix[4] =
			static_cast<float>(ui->doubleSpinBox11->value());
		params_->ccm_matrix[5] =
			static_cast<float>(ui->doubleSpinBox12->value());

		params_->ccm_matrix[6] =
			static_cast<float>(ui->doubleSpinBox20->value());
		params_->ccm_matrix[7] =
			static_cast<float>(ui->doubleSpinBox21->value());
		params_->ccm_matrix[8] =
			static_cast<float>(ui->doubleSpinBox22->value());
	}

	// 调用基类 accept 关闭窗口并返回 QDialog::Accepted
	QDialog::accept();
}