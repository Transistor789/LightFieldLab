#ifndef DIALOGWBGAINS_H
#define DIALOGWBGAINS_H

#include <QDialog>
#include <vector>

namespace Ui {
class DialogWBGains;
}

class DialogWBGains : public QDialog {
	Q_OBJECT

public:
	// 构造函数直接绑定引用
	explicit DialogWBGains(std::vector<float> &data, QWidget *parent = nullptr);
	~DialogWBGains();

protected:
	// 点击 OK 时保存
	void accept() override;

private:
	// 辅助函数：加载数据到 UI
	void loadDataToUI();

private:
	Ui::DialogWBGains *ui;
	std::vector<float> &wbGains; // 引用成员：直接指向外部数据
};

#endif // DIALOGWBGAINS_H