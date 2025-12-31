#ifndef DIALOGCCM_H
#define DIALOGCCM_H

#include <QDialog>
#include <vector>

namespace Ui {
class DialogCCM;
}

class DialogCCM : public QDialog {
	Q_OBJECT

public:
	// 构造时直接绑定数据引用
	explicit DialogCCM(std::vector<float> &data, QWidget *parent = nullptr);
	~DialogCCM();

protected:
	void accept() override;

private:
	// 将 SpinBox 控件批量加载数据到 UI
	void loadDataToUI();
	// 将 UI 数据批量保存回 Vector
	void saveDataToVector();

private:
	Ui::DialogCCM *ui;
	std::vector<float> &ccmMatrix; // 引用成员：直接指向外部数据
};

#endif // DIALOGCCM_H