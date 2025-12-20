#ifndef LFCONTROLLER_H
#define LFCONTROLLER_H

#include "lfcalibrate.h"
#include "lfcapture.h"
#include "lfdata.h"
#include "lfdisp.h"
#include "lfio.h"
#include "lfrefocus.h"
#include "lfresample.h"
#include "lfsr.h"

#include <future>
#include <memory>
#include <thread>

class LFController {
public:
	explicit LFController();
	~LFController();

	LfPtr lf;
	void update();
	void captureTask();
	void resampleTask();
	std::future<LfPtr> load_lf(const std::string &path);
	std::future<cv::Mat> refocus(float alpha, int crop);
	std::future<cv::Mat> super_res(const cv::Mat &img);

	bool exit;

private:
	std::unique_ptr<LFIO> load;
	std::unique_ptr<LFCalibrate> cal;

	std::unique_ptr<LFRefocus> ref;
	std::unique_ptr<LFSuperRes> sr;
	std::unique_ptr<LFDepth> dep;

	std::unique_ptr<LFCapture> cap;
	std::unique_ptr<LFResample> rsp;

	std::thread cap_thread;
	std::thread rsp_thread;
	std::mutex mtx;
	std::condition_variable cv;
	std::queue<cv::Mat> data_queue;
};

#endif