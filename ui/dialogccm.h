#ifndef DIALOGCCM_H
#define DIALOGCCM_H

#include <QDialog>

namespace Ui {
class DialogCCM;
}

class DialogCCM : public QDialog {
	Q_OBJECT

public:
	explicit DialogCCM(QWidget *parent = nullptr);
	~DialogCCM();

private:
	Ui::DialogCCM *ui;
};

#endif // DIALOGCCM_H
