#ifndef LFPROCESSOR_H
#define LFPROCESSOR_H

#include "lfdata.h"
#include "lfload.h"
#include "lfrefocus.h"
#include "lfsuperres.h"

class LFProcessor : public QObject {
	Q_OBJECT
public:
	enum WorkerType {
		LOAD = 0,
		REFOCUS = 1,
		SUPERRES = 2, // 示例：可扩展其他类型
		// DEPTH_ESTIMATION,
		WORKER_COUNT // 自动计数
	};
	explicit LFProcessor(QObject* parent = nullptr);
	~LFProcessor();

	void toFLoat(LightFieldPtr ptr);
	template <typename T>
	T* initWorker(WorkerType type);

	cv::Mat sai;
	LightFieldPtr lf, lf_float;
	QString lensletImagePath, whiteImagePath;
	bool isRgb = false, isGpu = false;
	int sai_row = 8, sai_col = 8, crop = 0;
	float alpha = 1.0;

	LFLoad* pLoad;
	LFRefocus* pRefocus;
	LFSuperres* pSuperres;
	QThread* threads[WORKER_COUNT];

public slots:
	void printThreadId();
	void onLFUpdated(const LightFieldPtr& src);
	void onGpuSliderValueChanged(int value);
	void onSRButtonClicked();

signals:
	void updateSAI(const cv::Mat& image);
	void updateLF_float(const LightFieldPtr& ptr);
	void updateLF_uchar(const LightFieldPtr& ptr);
	void requestUpsampleWithMat(const cv::Mat& src);
	void requestUpsampleWithPos(int row, int col);
	// void requestUpsample(const cv::Mat& src);
};

#endif