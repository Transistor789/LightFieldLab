#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "lfdata.h"

#include <memory>
#include <opencv2/dnn_superres.hpp>

class LFSuperRes {
public:
	LFSuperRes() = default;

	int type() const { return _type; }
	double scale() const { return _scale; }
	void setGpu(bool isGpu) { _isGpu = isGpu; }
	void setType(int index);
	void setScale(int index);
	void onUpdateLF(const LightFieldPtr &ptr) { lf = ptr; }
	void loadModel();
	int upsample(const cv::Mat &src, cv::Mat &dst);
	int upsample_single(int row, int col, cv::Mat &dst);
	void upsample_multiple();

	LightFieldPtr lf;

private:
	enum {
		NEAREST = 0,
		LINEAR = 1,
		CUBIC = 2,
		LANCZOS = 3,
		EDSR = 4,
		ESPCN = 5,
		FSRCNN = 6,
		TYPE_COUNT
	};

	bool _isGpu = false;
	double _scale = 2.0;
	int _type = NEAREST;
	cv::Mat _data;
	cv::Mat _input, _output;
	std::string _modelPath = "input/opencv_srmodel/";

	cv::dnn_superres::DnnSuperResImpl _sr;
};

#endif