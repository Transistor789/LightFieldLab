#ifndef LFDEPTH_
#define LFDEPTH_

#include "lfdata.h"

#include <opencv2/opencv.hpp>

class LFDepth {
public:
	explicit LFDepth();
	void update(const LfPtr &ptr);
	LfPtr lf;
};

#endif