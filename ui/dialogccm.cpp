#include "dialogccm.h"

#include "ui_dialogccm.h"

DialogCCM::DialogCCM(QWidget *parent) : QDialog(parent), ui(new Ui::DialogCCM) {
	ui->setupUi(this);
}

DialogCCM::~DialogCCM() { delete ui; }
