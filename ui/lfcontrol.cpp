#include "lfcontrol.h"

#include "config.h"
#include "logger.h"
#include "utils.h"
#include "widgetimage.h"

#include <QThreadPool>
#include <opencv2/highgui.hpp>

LFControl::LFControl(QObject *parent) : QObject(parent) {
	exit = false;
	isCapturing = false;
	isProcessing = false;

	cal = std::make_unique<LFCalibrate>();
	ref = std::make_unique<LFRefocus>();
	sr = std::make_unique<LFSuperRes>();
	dep = std::make_unique<LFDisp>();
	cap = std::make_unique<LFCapture>();
	isp = std::make_unique<LFIsp<uint16_t>>();

	cap_thread = std::thread(&LFControl::captureTask, this);
	isp_thread = std::thread(&LFControl::processTask, this);
}

LFControl::~LFControl() {
	stopAll();

	if (cap_thread.joinable()) {
		cap_thread.join();
	}

	if (isp_thread.joinable()) {
		isp_thread.join();
	}
}

void LFControl::stopAll() {
	isCapturing = false;
	isProcessing = false;

	// 设置退出标志
	exit = true;

	m_queueCv.notify_all();
}

void LFControl::setCapturing(bool active) {
	bool wasCapturing = isCapturing.load();

	// 状态未改变则直接返回
	if (wasCapturing == active)
		return;

	isCapturing.store(active);

	if (active) {
		LOG_INFO("Capture Started");
	} else {
		LOG_INFO("Capture Paused");
		// 可选：停止采集时，通常不需要唤醒谁，因为 captureTask 是轮询的
	}
}

void LFControl::setProcessing(bool active) {
	bool wasProcessing = isProcessing.load();

	// 状态未改变则直接返回
	if (wasProcessing == active)
		return;

	isProcessing.store(active);

	if (active) {
		LOG_INFO("Processing Resumed");
		// 【关键点】！！！
		// 消费者线程可能正卡在 cv.wait() 等待。
		// 如果不 notify，即使 isProcessing 变成了
		// true，线程也不会醒来检查条件， 只能等到下一帧数据入队时才会被唤醒。
		// 强制唤醒它，让它立即检查 (data_queue && isProcessing)
		m_queueCv.notify_all();
	} else {
		LOG_INFO("Processing Paused");
	}
}

void LFControl::captureTask() {
	// 检查硬件指针安全性
	if (!cap)
		return;

	while (!exit) { // atomic read，线程安全
		if (!isCapturing) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		cv::Mat img = cap->getFrame();

		if (img.empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		{
			std::lock_guard<std::mutex> lock(m_queueMtx);
			if (data_queue.size() > 5) {
				data_queue.pop();
			}
			data_queue.push(std::move(img));
		}

		m_queueCv.notify_one();
	}
}

void LFControl::processTask() {
	while (!exit) {
		cv::Mat img;
		{
			std::unique_lock<std::mutex> lock(m_queueMtx);

			m_queueCv.wait(lock, [this] {
				return exit.load()
					   || (!data_queue.empty() && isProcessing.load());
			});

			if (exit) {
				return;
			}

			if (data_queue.empty() || !isProcessing)
				continue;

			img = std::move(data_queue.front());
			data_queue.pop();
		}
	}
}

void LFControl::read_sai(const QString &path) {
	runAsync(
		[this, path] {
			// 1. 耗时操作
			LfPtr tempLF = LFIO::read_sai(path.toStdString());

			// 2. 加锁赋值
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				lf = tempLF;
			}

			// emit saiReady(cvMatToQImage(lf->getCenter()));
			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));
			// LOG_INFO("Sub-aperture images loaded");
		},
		"Loading sub-aperture images");
}

void LFControl::read_img(const QString &path, bool isWhite) {
	runAsync(
		[this, path, isWhite] {
			// 1. 耗时操作 (无锁)
			auto img = io->read_image(path.toStdString());

			// 2. 赋值 (加锁)
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				if (!isWhite) {
					lfraw = img;
					emit imageReady(ImageType::LFP, cvMatToQImage(lfraw));
				} else {
					white = img;
					emit imageReady(ImageType::White, cvMatToQImage(white));
				}
			}
		},
		isWhite ? "Loading white image"
				: "Loading light field image"); // 传入任务名用于报错
}

void LFControl::calibrate() {
	runAsync(
		[this] {
			cal->setImage(white);
			cal->run(cal_cca, cal_save);
			cal->computeSliceMaps(9);
			cal->computeDehexMaps();
			isp->maps.slice = cal->getSliceMaps();
			isp->maps.dehex = cal->getDehexMaps();
		},
		"Calibrating");
}

void LFControl::process() {
	runAsync(
		[this] {
			isp->set_lf_img(lfraw);
			isp->set_white_img(white);
		},
		"Processing");
}

QImage LFControl::cvMatToQImage(const cv::Mat &inMat) {
	// 1. 处理数据类型 (如果是 float 或 16bit，先转为 8bit 显示用)
	cv::Mat temp;
	if (inMat.depth() == CV_32F) {
		// 归一化 0.0-1.0 -> 0-255
		inMat.convertTo(temp, CV_8UC(inMat.channels()), 255.0);
	} else if (inMat.depth() == CV_16U) {
		// 16bit -> 8bit (简单压缩，或者你可以做更复杂的 Tone Mapping)
		inMat.convertTo(temp, CV_8UC(inMat.channels()), 255.0 / 1023.0);
	} else {
		temp = inMat; // 已经是 8bit 就不拷贝
	}

	// 2. 处理通道顺序 (OpenCV 是 BGR，Qt 是 RGB)
	if (temp.channels() == 3) {
		cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);
		return QImage((const uchar *)temp.data, temp.cols, temp.rows, temp.step,
					  QImage::Format_RGB888)
			.copy();
	} else if (temp.channels() == 1) {
		// 灰度图
		return QImage((const uchar *)temp.data, temp.cols, temp.rows, temp.step,
					  QImage::Format_Grayscale8)
			.copy();
	}

	return QImage();
}