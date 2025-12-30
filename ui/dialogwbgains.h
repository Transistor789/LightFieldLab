#ifndef DIALOGWHITEGAIN_H
#define DIALOGWHITEGAIN_H

#include <QDialog>
// 前置声明，减少编译依赖
struct LFParamsSource;

namespace Ui {
class DialogWBGains;
}

class DialogWBGains : public QDialog {
	Q_OBJECT

public:
	explicit DialogWBGains(QWidget *parent = nullptr);
	~DialogWBGains();

	// 核心接口：注入参数指针
	void setupParams(LFParamsSource *params);

protected:
	// 覆写 accept 函数 (当用户点击 OK 时触发)
	void accept() override;

private:
	Ui::DialogWBGains *ui;
	LFParamsSource *m_params = nullptr; // 保存指针
};

#endif // DIALOGWHITEGAIN_H