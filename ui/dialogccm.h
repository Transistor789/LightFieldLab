#ifndef DIALOGCCM_H
#define DIALOGCCM_H

#include "lfparams.h"

#include <QDialog>


namespace Ui {
class DialogCCM;
}

class DialogCCM : public QDialog {
	Q_OBJECT

public:
	explicit DialogCCM(QWidget *parent = nullptr);
	~DialogCCM();

	// 传入参数指针并初始化UI
	void setupParams(LFParamsSource *params);

protected:
	// 重写 accept，点击 OK 时保存数据
	void accept() override;

private slots:
	// 如果需要在界面打开后重置/刷新数据，可以使用此槽
	void updateUI();

private:
	Ui::DialogCCM *ui;
	LFParamsSource *params_ = nullptr; // 初始化为空指针
};

#endif // DIALOGCCM_H