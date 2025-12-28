#ifndef DIALOGWHITEGAIN_H
#define DIALOGWHITEGAIN_H

#include <QDialog>
#include <vector>

namespace Ui {
class DialogWBGains;
}

class DialogWBGains : public QDialog {
	Q_OBJECT

public:
	explicit DialogWBGains(QWidget *parent = nullptr);
	~DialogWBGains();

	std::vector<double> getGains();
	void set(const std::vector<double> &gains);

private:
	Ui::DialogWBGains *ui;
};

#endif // DIALOGWHITEGAIN_H
