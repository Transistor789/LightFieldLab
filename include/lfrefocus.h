#ifndef LFREFOCUS_H
#define LFREFOCUS_H

#include "lfdata.h"

class LFRefocus : public QObject {
	Q_OBJECT
public:
	explicit LFRefocus(QObject* parent = nullptr);
	void init(const LightFieldPtr& ptr);
	bool getGpu() const { return _isGpu; }

	LightFieldPtr lf;
public slots:
	void printThreadId();
	void setGpu(bool isGPU);
	void refocus(float alpha, int crop);
	void onUpdateLF(const LightFieldPtr& ptr);

signals:
	void finished(const cv::Mat& image);

private:
	bool _isGpu = false;
	int _views, _len, _center, _type;
	cv::Mat _xmap, _ymap, _refocusedImage;
	cv::UMat _xmap_gpu, _ymap_gpu;
	cv::Size _size;
};

#endif