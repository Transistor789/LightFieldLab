#include "dialogwbgains.h"

#include "ui_dialogwbgains.h"

DialogWBGains::DialogWBGains(QWidget *parent)
	: QDialog(parent), ui(new Ui::DialogWBGains) {
	ui->setupUi(this);
}

DialogWBGains::~DialogWBGains() { delete ui; }

std::vector<double> DialogWBGains::getGains() {
	return {ui->gain1->value(), ui->gain2->value(), ui->gain3->value(),
			ui->gain4->value()};
}

void DialogWBGains::set(const std::vector<double> &gains) {
	ui->gain1->setValue(gains[0]);
	ui->gain2->setValue(gains[1]);
	ui->gain3->setValue(gains[2]);
	ui->gain4->setValue(gains[3]);
}
