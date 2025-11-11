#ifndef LFPROCESSOR_H
#define LFPROCESSOR_H

#include "lfdata.h"
#include "qlfapi.h"

class LFProcessor : public QObject {
	Q_OBJECT
public:
	enum {
		LOAD = 0,
		REFOCUS = 1,
		SUPERRES = 2, // 示例：可扩展其他类型
		// DEPTH_ESTIMATION,
		WORKER_COUNT // 自动计数
	};
	explicit LFProcessor(QObject *parent = nullptr);
	~LFProcessor() override;

	template <typename T>
	T *initWorker(int type);

	cv::Mat sai;
	LightFieldPtr lf;
	QString lensletImagePath, whiteImagePath;
	bool isRgb = GRAY, isGpu = CPU;
	int sai_row = 8, sai_col = 8, crop = 0;
	float alpha = 1.0;

	QLFLoad *qload;
	QLFRefocus *qrefocus;
	QLFSuperRes *qsuperres;
	QThread *threads[WORKER_COUNT];

public slots:
	static void printThreadId();
	void onLFUpdated(const LightFieldPtr &src);
	void onGpuSliderValueChanged(int value);
	void onSRButtonClicked();

signals:
	void updateSAI(cv::Mat);
	void sendLfPtr(const LightFieldPtr &);
	void requestUpsampleWithMat(cv::Mat);
	void requestUpsampleWithPos(int, int);
	// void requestUpsample(const cv::Mat& src);
};

#endif