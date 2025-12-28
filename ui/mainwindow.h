#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "lfcontrol.h"

#include <QMainWindow>
#include <memory>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = nullptr);
	~MainWindow();

private:
	Ui::MainWindow *ui;

	std::unique_ptr<LFControl> ctrl;
};

#endif // MAINWINDOW_H
