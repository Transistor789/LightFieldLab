#ifndef LFRESAMPLE_H
#define LFRESAMPLE_H

#include "lfdata.h"

class LFResample {
public:
	explicit LFResample();
	LfPtr run(const cv::Mat &);

	LfPtr lf;
};

#endif