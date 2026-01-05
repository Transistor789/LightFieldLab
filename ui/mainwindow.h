#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "lfcontrol.h"

#include <QElapsedTimer>
#include <QMainWindow>
#include <QTimer>
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
	void initStatusBar(); // 初始化状态栏的专用函数

	// --- 状态栏控件指针 ---

	// 1. 交互信息区 (Interactive Info)
	QLabel *lblCoords; // 坐标: X: 1024 Y: 768
	QLabel *lblRGB;	   // 颜色: R: 255 G: 255 B: 255
	QLabel *lblZoom;   // 缩放: 150%

	// 2. 性能指标区 (Performance Metrics)
	QLabel *lblResolution; // 分辨率: 1920x1080
	QLabel *lblFPS;		   // 帧率: FPS: 30.1

private slots:
	void updateMouseInfo(int x, int y, const QColor &color, double scale, int w,
						 int h);
	void updateZoomInfo(double scale);
	void updateFPS(double fps);

private:
	Ui::MainWindow *ui;

	std::unique_ptr<LFControl> ctrl;
	QTimer *m_fpsTimer; // 1秒定时器
	QElapsedTimer m_fpsTimeCalculator;
	int m_uiFrameCount = 0; // UI 渲染计数器 (仅主线程访问)
};

#endif // MAINWINDOW_H
