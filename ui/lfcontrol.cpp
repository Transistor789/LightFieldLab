#include "lfcontrol.h"

#include "config.h"
#include "lfdata.h"
#include "lfparams.h"
#include "logger.h"
#include "utils.h"
#include "widgetimage.h"

#include <QThreadPool>
#include <format>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <qcontainerfwd.h>
#include <qtmetamacros.h>

LFControl::LFControl(QObject *parent) : QObject(parent) {
	exit = false;
	isCapturing = false;
	isProcessing = false;

	cal = std::make_unique<LFCalibrate>();
	ref = std::make_unique<LFRefocus>();
	sr = std::make_unique<LFSuperRes>();
	dep = std::make_unique<LFDisp>();
	cap = std::make_unique<LFCapture>();
	isp = std::make_unique<LFIsp>();

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

void LFControl::readSAI(const QString &path) {
	runAsync(
		[this, path] {
			// 1. 耗时操作
			LfPtr tempLF = LFIO::readSAI(path.toStdString());

			// 2. 加锁赋值
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				lf = tempLF;
			}

			// emit saiReady(cvMatToQImage(lf->getCenter()));
			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));
			params.source.pathSAI = path.toStdString();
			emit paramsChanged();
			// LOG_INFO("Sub-aperture images loaded");
		},
		"Loading sub-aperture images");
}

void LFControl::readImage(const QString &path, bool isWhite) {
	runAsync(
		[this, path, isWhite] {
			// 1. 耗时操作 (无锁)
			auto img =
				io->readImage(path.toStdString(), &Config::Get().img_meta());

			// 2. 赋值 (加锁)
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				if (!isWhite) {
					lfraw = img;
					isp->set_config(Config::Get().img_meta());
					params.source = isp->get_config();
					params.source.pathLFP = path.toStdString();
					emit imageReady(ImageType::LFP, cvMatToQImage(lfraw));
				} else {
					white = img;
					isp->set_white_img(white.clone());
					params.source.pathWhite = path.toStdString();
					emit imageReady(ImageType::White, cvMatToQImage(white));
				}
				params.source.height = img.size().height;
				params.source.width = img.size().width;
			}
			emit paramsChanged();
		},
		isWhite ? "Loading white image"
				: "Loading light field image"); // 传入任务名用于报错
}

void LFControl::readExtractLUT(const QString &path) {
	runAsync(
		[this, path] {
			LFIO::loadLookUpTables(path.toStdString(), isp->maps.extract,
								   params.calibrate.views);
			params.source.pathExtract = path.toStdString();
			emit paramsChanged();
		},
		"Load extract-lut");
}

void LFControl::readDehexLUT(const QString &path) {
	runAsync(
		[this, path] {
			int _;
			LFIO::loadLookUpTables(path.toStdString(), isp->maps.dehex, _);
			params.source.pathDehex = path.toStdString();
			emit paramsChanged();
		},
		"Load dehex-lut");
}

void LFControl::calibrate() {
	runAsync(
		[this] {
			cal->setImage(white);
			auto points =
				cal->run(params.calibrate.useCCA, params.calibrate.saveLUT,
						 params.source.bayer != BayerPattern::NONE,
						 params.source.bitDepth);
			cv::Mat draw =
				draw_points(white, points, "", 2, cv::Scalar(0), false);

			emit imageReady(ImageType::White, cvMatToQImage(draw));
			emit paramsChanged();
		},
		"Calibrating");
}

void LFControl::genLUT() {
	runAsync(
		[this] {
			isp->maps.extract = cal->computeSliceMaps(params.calibrate.views);
			isp->maps.dehex = cal->computeDehexMaps();
			if (params.calibrate.saveLUT) {
				io->saveLookUpTables(
					std::format("data/calibration/lut_extract_{}.bin",
								params.calibrate.views),
					isp->maps.extract, params.calibrate.views);
				io->saveLookUpTables("data/calibration/lut_dehex.bin",
									 isp->maps.dehex, 1);
			}
			emit paramsChanged();
		},
		"Calibrating");
}

void LFControl::process() {
	runAsync(
		[this] {
			isp->set_lf_img(lfraw.clone());
			if (params.isp.enableDPC) {
				isp->dpc_fast();
			}
			if (params.isp.enableBLC) {
				isp->blc_fast();
			}
			if (params.isp.enableLSC && params.isp.enableAWB) {
				isp->lsc_awb_fused_fast();
			} else if (params.isp.enableLSC) {
				isp->lsc_fast();
			} else if (params.isp.enableAWB) {
				isp->awb_fast();
			}
			if (params.isp.enableDemosaic) {
				isp->demosaic();
			}
			if (params.isp.enableCCM) {
				isp->ccm_fast();
			}
			emit imageReady(ImageType::LFP, cvMatToQImage(isp->getResult()));

			if (params.isp.enableExtract) {
				isp->resample(params.isp.enableDehex);
			}
			if (params.isp.enableColorEq) {
				isp->color_equalize();
			}
			lf = std::make_shared<LFData>(isp->getSAIS());
			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));
			emit imageReady(ImageType::SAI, cvMatToQImage(lf->getCenter()));
			emit paramsChanged();
		},
		"ISP");
}

void LFControl::fast_preview() {
	runAsync(
		[this] {
			isp->set_lf_img(lfraw.clone());
			auto img = isp->preview().getPreviewResult();
			emit imageReady(ImageType::LFP, cvMatToQImage(img));
			emit paramsChanged();
		},
		"Fast preview");
}
void LFControl::refocus() {
	runAsync(
		[this] {
			ref->setLF(lf);
			auto img = ref->refocus(params.refocus.alpha, params.refocus.crop);
			emit imageReady(ImageType::Refocus, cvMatToQImage(img));
			emit paramsChanged();
		},
		"Refocus");
}
void LFControl::upsample() {
	runAsync(
		[this] {
			auto img = sr->upsample(lf->getCenter());
			emit imageReady(ImageType::SR, cvMatToQImage(img));
			emit paramsChanged();
		},
		"Super resolution");
}
void LFControl::depth() {
	runAsync(
		[this] {
			auto img = dep->depth(lf->data);
			if (params.de.color == LFParamsDE::Color::Gray) {
				emit imageReady(ImageType::Depth,
								cvMatToQImage(dep->getGrayVisual()));
			} else if (params.de.color == LFParamsDE::Color::Jet) {
				emit imageReady(ImageType::Depth,
								cvMatToQImage(dep->getJetVisual()));
			} else {
				emit imageReady(ImageType::Depth,
								cvMatToQImage(dep->getPlasmaVisual()));
			}

			emit paramsChanged();
		},
		"Depth estimation");
}

QImage LFControl::cvMatToQImage(const cv::Mat &inMat) {
	if (inMat.empty())
		return QImage();

	// 1. 处理 3通道 8位 (彩色) - 最常见情况
	if (inMat.type() == CV_8UC3) {
		QImage image((const uchar *)inMat.data, inMat.cols, inMat.rows,
					 inMat.step, QImage::Format_BGR888);
		return image.copy();
	}

	// 2. 处理 1通道 8位 (灰度)
	else if (inMat.type() == CV_8UC1) {
		QImage image((const uchar *)inMat.data, inMat.cols, inMat.rows,
					 inMat.step, QImage::Format_Grayscale8);
		return image.copy();
	}

	// 3. 处理高位深或其他格式 (10/12/16-bit, float)
	else {
		// std::cerr << "[Info] Converting high-bit/float image for display..."
		// << std::endl;

		cv::Mat temp;
		double alpha = 1.0;
		double beta = 0.0;

		// === 核心修改开始：根据 bitDepth 计算缩放系数 ===

		// 情况 A: 浮点型 (CV_32F / CV_64F)
		// 通常浮点型已经是 0.0-1.0 (归一化过的)，可以直接乘 255
		// 但如果算法产生的浮点图是 0-1023 的范围，这里可能需要调整
		if (inMat.depth() == CV_32F || inMat.depth() == CV_64F) {
			// 这里假设浮点图是 0.0-1.0 的标准格式
			// 如果你的浮点图范围是 0-1023，这里的逻辑需要改成下面 16U 的逻辑
			alpha = 255.0;
		}

		// 情况 B: 整数型 (CV_16U / CV_16S)
		// 这里必须用到 params.source.bitDepth
		else if (inMat.depth() == CV_16U || inMat.depth() == CV_16S) {
			int validBits = params.source.bitDepth;

			// 防御性编程：如果用户没设置或者乱设置，给个默认值
			if (validBits <= 0)
				validBits = 16;
			if (validBits > 16)
				validBits = 16;

			double maxVal = (1 << validBits) - 1.0;

			// 归一化系数：将 [0, maxVal] 映射到 [0, 255]
			alpha = 255.0 / maxVal;
		}
		inMat.convertTo(temp, CV_8U, alpha, beta);

		if (temp.channels() == 1) {
			cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
		} else if (temp.channels() == 3) {
			// convertTo 不会改变通道数，如果原来是 RGB/BGR 的 16位图，现在是
			// 8位 这里不需要做额外操作，除非涉及到 RGB/BGR 互换， 假设已经是
			// BGR 顺序 (OpenCV 默认)
		}

		// 递归调用自己 (此时 temp 已经是 CV_8UC3 或 CV_8UC1，会进入上面两个 if)
		return cvMatToQImage(temp);
	}
}