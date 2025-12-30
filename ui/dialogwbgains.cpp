#include "dialogwbgains.h"

#include "lfparams.h" // 必须包含完整定义
#include "ui_dialogwbgains.h"

DialogWBGains::DialogWBGains(QWidget *parent)
	: QDialog(parent), ui(new Ui::DialogWBGains) {
	ui->setupUi(this);
	// 这里不需要做任何 connect，因为我们只在打开和关闭时同步数据
}

DialogWBGains::~DialogWBGains() { delete ui; }

// 【方向 1】: Params -> UI (初始化显示)
void DialogWBGains::setupParams(LFParamsSource *params) {
	m_params = params;
	if (!m_params)
		return;

	// 假设你有 4 个 SpinBox 对应 RGGB
	// 注意：要做好越界检查，防止 vector 为空崩掉
	if (m_params->awb_gains.size() >= 4) {
		ui->gain1->setValue(m_params->awb_gains[0]);
		ui->gain2->setValue(m_params->awb_gains[1]);
		ui->gain3->setValue(m_params->awb_gains[2]);
		ui->gain4->setValue(m_params->awb_gains[3]);
	}
}

// 【方向 2】: UI -> Params (保存数据)
void DialogWBGains::accept() {
	if (m_params) {
		// 只有用户点击了 OK，我们才把界面上的值写回参数
		// 这样如果用户点 Cancel，参数就不会被弄乱，无需额外的回滚逻辑

		// 确保 vector 大小足够
		if (m_params->awb_gains.size() < 4) {
			m_params->awb_gains.resize(4, 1.0f);
		}

		m_params->awb_gains[0] = static_cast<float>(ui->gain1->value());
		m_params->awb_gains[1] = static_cast<float>(ui->gain2->value());
		m_params->awb_gains[2] = static_cast<float>(ui->gain3->value());
		m_params->awb_gains[3] = static_cast<float>(ui->gain4->value());
	}

	// 调用父类 accept，关闭窗口并返回 QDialog::Accepted
	QDialog::accept();
}