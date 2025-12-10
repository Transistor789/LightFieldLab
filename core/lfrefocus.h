#ifndef LFREFOCUS_H
#define LFREFOCUS_H

#include "lfdata.h"

class LFRefocus {
public:
	LFRefocus() = default;
	~LFRefocus() = default;

	void init(const LfPtr &ptr);
	cv::Mat refocus(float alpha, int crop = 0);
	void update(const LfPtr &ptr);

	LfPtr lf;

private:
	int _views, _len, _center, _type;

	cv::Size _size;
	cv::Mat _xmap, _ymap;
};

#endif