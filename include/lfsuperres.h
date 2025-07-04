#ifndef LFSUPERRES_H
#define LFSUPERRES_H

#include "lfdata.h"

class LFSuperres : public QObject {
	Q_OBJECT
public:
	explicit LFSuperres(QObject *parent = nullptr);
	enum SR_type {
		NEAREST = 0,
		LINEAR = 1,
		CUBIC = 2,
		LANCZOS = 3,
		EDSR = 4,
		ESPCN = 5,
		FSRCNN = 6,
		TYPE_COUNT
	};
	SR_type type() const { return _type; }
	double scale() const { return _scale; }
	void setGpu(bool isGpu);

	LightFieldPtr lf, lf_float;

public slots:
	void printThreadId();
	void setType(int index);
	void setScale(int index);
	void onUpdateLF(const LightFieldPtr &ptr);
	void loadModel();
	void upsample_single(const cv::Mat &src);
	void upsample_single(int row, int col);
	void upsample_multiple();

signals:
	void finished(const cv::Mat &image);

private:
	bool _isGpu = false;
	double _scale = 2.0;

	SR_type _type = NEAREST;

	cv::Mat _data;
	cv::Mat _input, _output;
	cv::UMat _input_gpu, _output_gpu;
	std::string _modelPath = "input/opencv_srmodel/";

	cv::dnn_superres::DnnSuperResImpl _sr;
};

#endif