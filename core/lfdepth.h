#ifndef LFDEPTH_
#define LFDEPTH_

#include "lfdata.h"

#include <opencv2/opencv.hpp>

class LFDisp {
public:
	explicit LFDisp();
	void update(const LfPtr &ptr);
	LfPtr lf;
};

#endif