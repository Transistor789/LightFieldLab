#ifndef CONTROL_H
#define CONTROL_H

#include "lfcalibrate.h"
#include "lfcapture.h"
#include "lfdepth.h"
#include "lfio.h"
#include "lfisp.h"
#include "lfparams.h"
#include "lfrefocus.h"
#include "lfsr.h"
#include "logger.h"
#include "utils.h"

#include <QImage>
#include <QMetaType>
#include <QMutex>
#include <QObject>
#include <QString>
#include <QtConcurrent/QtConcurrent>
#include <format>
#include <qimage.h>
#include <qtmetamacros.h>
#include <queue>

enum class ImageType;

class LFControl : public QObject {
	Q_OBJECT
public:
	explicit LFControl(QObject *parent = nullptr);
	~LFControl();

	void captureTask();
	void processTask();
	void stopAll();

public slots:
	void setCapturing(bool active);
	void setProcessing(bool active);
	bool getIsCapturing() const { return params.dynamic.isCapturing.load(); }
	bool getIsProcessing() const { return params.dynamic.isProcessing.load(); }

	void readSAI(const QString &path);
	void readStandardImage(const QString &path);
	void readLFP(const QString &path);
	void readWhite(const QString &path);
	void readExtractLUT(const QString &path);
	void readDehexLUT(const QString &path);

	void calibrate();
	void genLUT();
	void process();
	void fast_preview();
	void detectCamera();
	void updateSAI(int row, int col);
	void play();
	void refocus();
	void processAllInFocus();
	void upsample();
	void depth();

signals:
	void updateSAI(const cv::Mat &img);
	void imageReady(ImageType type, const QImage &img);
	void paramsChanged();

private:
	QImage cvMatToQImage(const cv::Mat &inMat);

	template <typename Func>
	void runAsync(Func &&task, const QString &taskName) {
		QThreadPool::globalInstance()->start(
			QRunnable::create([this, task, taskName]() mutable {
				// 1. 创建计时器 (构造时自动 start)
				Timer timer;

				try {
					// 2. 执行任务
					task();

					// 3. 任务结束，停止计时
					timer.stop();
					double costMs = timer.elapsed_ms();

					// 4. 打印日志 (包含耗时)
					// {:.2f} 保留两位小数，看起来更整洁
					LOG_INFO(std::format("{} finished. (Time: {:.2f} ms)",
										 taskName.toStdString(), costMs));

					// 5. (进阶) 如果你在 UI
					// 上做了状态栏显示，可以在这里发射信号 emit
					// taskFinished(taskName, costMs);

				} catch (const std::exception &e) {
					LOG_ERROR(std::format("{} failed: {}",
										  taskName.toStdString(), e.what()));

				} catch (...) {
					LOG_ERROR(std::format("{} failed: Unknown error",
										  taskName.toStdString()));
				}
			}));
	}

public:
	LFParams params;

private:
	std::shared_ptr<LFData> lf;
	cv::Mat lfraw, white;
	mutable std::mutex m_dataMtx; // 保护上述数据

	// --- 模块实例 ---
	std::unique_ptr<LFIO> io;
	std::unique_ptr<LFCalibrate> cal;
	std::unique_ptr<LFIsp> isp;
	std::unique_ptr<LFRefocus> ref;
	std::unique_ptr<LFSuperRes> sr;
	std::unique_ptr<LFDisp> dep;
	std::unique_ptr<LFCapture> cap;

	// --- 线程与同步 ---
	std::thread cap_thread;
	std::thread proc_thread;

	// 生产者-消费者队列
	std::queue<cv::Mat> data_queue;
	std::mutex m_queueMtx;
	std::condition_variable m_queueCv;
};

#endif