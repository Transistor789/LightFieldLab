#ifndef LFREFOCUS_H
#define LFREFOCUS_H

#include "lfdata.h"

#include <memory>

class LFRefocus {
public:
	LFRefocus() = default;
	~LFRefocus();

	void init(const LightFieldPtr &ptr);
	bool getGpu() const { return _isGpu; }
	void setGpu(bool isGPU);
	int refocus(cv::Mat &img, float alpha, int crop);
	int refocus_cpu(cv::Mat &img, float alpha, int crop);
	int refocus_gpu(cv::Mat &img, float alpha, int crop);
	void onUpdateLF(const LightFieldPtr &ptr);

	LightFieldPtr lf;

private:
	// struct metaData {};
	bool _isGpu = false;
	int _views, _len, _center, _type;

	cv::Size _size;
	cv::Mat _xmap, _ymap;
	cv::UMat _xmap_gpu, _ymap_gpu;
};

#endif