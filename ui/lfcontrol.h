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
	bool getIsCapturing() const { return isCapturing.load(); }
	bool getIsProcessing() const { return isProcessing.load(); }

	void readSAI(const QString &path);
	void readImage(const QString &path, bool isWhite);
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
		// 使用 QRunnable::create 包装任务
		QThreadPool::globalInstance()->start(
			QRunnable::create([this, task, taskName]() mutable {
				try {
					// 执行具体业务逻辑
					task();

					// 统一打成功日志 (可选，防止日志泛滥)
					LOG_INFO(
						std::format("{} finished.", taskName.toStdString()));
				} catch (const std::exception &e) {
					// 统一异常处理
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
	std::thread isp_thread;

	// 控制标志
	std::atomic<bool> exit;
	std::atomic<bool> isCapturing;
	std::atomic<bool> isProcessing;

	// 生产者-消费者队列
	std::queue<cv::Mat> data_queue;
	std::mutex m_queueMtx;
	std::condition_variable m_queueCv;
};

#endif