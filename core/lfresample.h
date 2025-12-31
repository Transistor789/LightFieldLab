#ifndef LFRESAMPLE_H
#define LFRESAMPLE_H

#include "lfdata.h"

class LFResample {
public:
	explicit LFResample();
	std::shared_ptr<LFData> run(const cv::Mat &);

	std::shared_ptr<LFData> lf;
};

#endif