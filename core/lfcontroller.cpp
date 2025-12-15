#include "lfcontroller.h"

#include "lfdata.h"

#include <filesystem>
#include <future>

namespace fs = std::filesystem;

LFController::LFController() : exit(false) {
	load = std::make_unique<LFIO>();
	cal = std::make_unique<LFCalibrate>();
	ref = std::make_unique<LFRefocus>();
	sr = std::make_unique<LFSuperRes>();
	dep = std::make_unique<LFDepth>();
	cap = std::make_unique<LFCapture>();

	cap_thread = std::thread(&LFController::captureTask, this);
	rsp_thread = std::thread(&LFController::resampleTask, this);
}

LFController::~LFController() {
	if (cap_thread.joinable()) {
		cap_thread.join();
	}
	if (rsp_thread.joinable()) {
		rsp_thread.join();
	}
}

void LFController::captureTask() {
	while (!exit) {
		cv::Mat img = cap->getFrame();
		{
			std::lock_guard<std::mutex> lock(mtx);
			data_queue.push(std::move(img));
		}

		cv.notify_one();
	}
}

void LFController::resampleTask() {
	while (!exit) {
		cv::Mat img;
		{
			std::unique_lock<std::mutex> lock(mtx);
			cv.wait(lock, [this] { return exit || !data_queue.empty(); });

			if (exit) {
				return;
			}

			img = std::move(data_queue.front());
			data_queue.pop();
		}
		LfPtr result = rsp->run(img);
	}
}

void LFController::update() {
	ref->update(lf);
	sr->update(lf);
	dep->update(lf);
}

std::future<LfPtr> LFController::load_lf(const std::string &path) {
	return std::async(std::launch::async, [this, path]() -> LfPtr {
		LfPtr lf;
		if (fs::is_directory(path)) {
			lf = load->read_sai(path, false);
		} else {
			cv::Mat img = load->read_image(path);
			lf = rsp->run(img);
		}
		return lf;
	});
}
std::future<cv::Mat> LFController::refocus(float alpha, int crop) {
	return std::async(std::launch::async, [this, alpha, crop]() -> cv::Mat {
		return ref->refocus(alpha, crop);
	});
}

std::future<cv::Mat> LFController::super_res(const cv::Mat &img) {
	return std::async(std::launch::async,
					  [this, img]() -> cv::Mat { return sr->upsample(img); });
}