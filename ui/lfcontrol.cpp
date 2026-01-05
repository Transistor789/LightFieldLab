#include "lfcontrol.h"

#include "config.h"
#include "lfdata.h"
#include "lfparams.h"
#include "logger.h"
#include "utils.h"
#include "widgetimage.h"

#include <QThreadPool>
#include <chrono>
#include <format>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <qcontainerfwd.h>
#include <qtmetamacros.h>
#include <thread>

LFControl::LFControl(QObject *parent) : QObject(parent) {
	cal = std::make_unique<LFCalibrate>();
	ref = std::make_unique<LFRefocus>();
	sr = std::make_unique<LFSuperRes>();
	dep = std::make_unique<LFDisp>();
	cap = std::make_unique<LFCapture>();
	isp = std::make_unique<LFIsp>();

	cap_thread = std::thread(&LFControl::captureTask, this);
	proc_thread = std::thread(&LFControl::processTask, this);
}

LFControl::~LFControl() {
	stopAll();

	if (cap_thread.joinable()) {
		cap_thread.join();
	}

	if (proc_thread.joinable()) {
		proc_thread.join();
	}
}

void LFControl::stopAll() {
	params.dynamic.isCapturing = false;
	params.dynamic.isProcessing = false;
	params.dynamic.exit = true;

	m_queueCv.notify_all();
}

void LFControl::setCapturing(bool active) {
	bool wasCapturing = params.dynamic.isCapturing.load();

	// 状态未改变则直接返回
	if (wasCapturing == active)
		return;

	params.dynamic.isCapturing.store(active);

	if (active) {
		LOG_INFO("Capture Started");
	} else {
		LOG_INFO("Capture Paused");
		// 可选：停止采集时，通常不需要唤醒谁，因为 captureTask 是轮询的
	}
}

void LFControl::setProcessing(bool active) {
	bool wasProcessing = params.dynamic.isProcessing.load();

	// 状态未改变则直接返回
	if (wasProcessing == active)
		return;

	params.dynamic.isProcessing.store(active);

	if (active) {
		LOG_INFO("Processing Resumed");
		// 【关键点】！！！
		// 消费者线程可能正卡在 cv.wait() 等待。
		// 如果不 notify，即使 params.dynamic.isProcessing 变成了
		// true，线程也不会醒来检查条件， 只能等到下一帧数据入队时才会被唤醒。
		// 强制唤醒它，让它立即检查 (data_queue && params.dynamic.isProcessing)
		m_queueCv.notify_all();
	} else {
		LOG_INFO("Processing Paused");
	}
}

void LFControl::captureTask() {
	if (!cap)
		return;

	while (!params.dynamic.exit.load()) {
		if (!params.dynamic.isCapturing.load()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		cv::Mat img = cap->getFrame();
		// cv::Mat img;
		LOG_INFO("Capturing ...");

		params.dynamic.capFrameCount++;
		if (img.empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			LOG_WARN("Image is empty!");
			continue;
		}

		{
			std::lock_guard<std::mutex> lock(m_queueMtx);
			if (data_queue.size() > 2) {
				data_queue.pop();
			}
			data_queue.push(std::move(img));
		}

		m_queueCv.notify_one();
	}
}

void LFControl::processTask() {
	while (!params.dynamic.exit) {
		cv::Mat img;
		{
			std::unique_lock<std::mutex> lock(m_queueMtx);

			m_queueCv.wait(lock, [this] {
				return params.dynamic.exit.load()
					   || (!data_queue.empty()
						   && params.dynamic.isProcessing.load());
			});

			LOG_INFO("Processing");
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			if (params.dynamic.exit) {
				return;
			}

			if (data_queue.empty() || !params.dynamic.isProcessing)
				continue;

			img = std::move(data_queue.front());
			data_queue.pop();

			isp->set_lf_img(img);
			isp->preview().resample(true);
			lf = std::make_shared<LFData>(isp->getSAIS());
			params.dynamic.procFrameCount++;
		}
	}
}

void LFControl::readSAI(const QString &path) {
	runAsync(
		[this, path] {
			// 1. 耗时操作
			std::shared_ptr<LFData> tempLF = LFIO::readSAI(path.toStdString());

			// 2. 加锁赋值
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				lf = tempLF;
			}

			// emit saiReady(cvMatToQImage(lf->getCenter()));
			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));
			params.path.sai = path.toStdString();
			params.isp.width = lf->width;
			params.isp.height = lf->height;
			params.isp.bitDepth = 8;
			params.isp.bayer = BayerPattern::NONE;
			params.sai.row = (lf->rows + 1) / 2;
			params.sai.col = (lf->cols + 1) / 2;
			params.sai.rows = lf->rows;
			params.sai.cols = lf->cols;

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
					params.isp = isp->get_config();
					params.path.lfp = path.toStdString();
					emit imageReady(ImageType::LFP, cvMatToQImage(lfraw));
				} else {
					white = img;
					isp->set_white_img(white.clone());
					params.path.white = path.toStdString();
					emit imageReady(ImageType::White, cvMatToQImage(white));
				}
				params.isp.height = img.size().height;
				params.isp.width = img.size().width;
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
			params.path.extractLUT = path.toStdString();
			emit paramsChanged();
		},
		"Load extract-lut");
}

void LFControl::readDehexLUT(const QString &path) {
	runAsync(
		[this, path] {
			int _;
			LFIO::loadLookUpTables(path.toStdString(), isp->maps.dehex, _);
			params.path.dehexLUT = path.toStdString();
			emit paramsChanged();
		},
		"Load dehex-lut");
}

void LFControl::calibrate() {
	runAsync(
		[this] {
			cal->setImage(white);
			auto points = cal->run(params.calibrate.useCCA,
								   params.isp.bayer != BayerPattern::NONE,
								   params.isp.bitDepth);
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
			emit paramsChanged();
		},
		"ISP");
}

void LFControl::detectCamera() {
	runAsync(
		[this] {
			params.dynamic.cameraID = cap->getAvailableCameras(3);
			emit paramsChanged();
		},
		"Detecting camera");
}

void LFControl::fast_preview() {
	runAsync(
		[this] {
			isp->set_lf_img(lfraw.clone());
			auto img = isp->preview().getPreviewResult();
			emit imageReady(ImageType::LFP, cvMatToQImage(img));
			if (params.isp.enableExtract) {
				isp->resample(params.isp.enableDehex);
			}
			if (params.isp.enableColorEq) {
				isp->color_equalize();
			}
			lf = std::make_shared<LFData>(isp->getSAIS());
			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));
			emit paramsChanged();
		},
		"Fast preview");
}

void LFControl::updateSAI(int row, int col) {
	runAsync(
		[this, row, col] {
			emit imageReady(ImageType::Center,
							cvMatToQImage(lf->getSAI(row - 1, col - 1)));
			emit paramsChanged();
		},
		"SAI updated");
}

void LFControl::play() {
	if (!params.sai.isPlaying) {
		return;
	}
	runAsync(
		[this] {
			LOG_INFO("Playing...");

			// 缓存一下边界，使代码更简洁
			// 假设 params.sai.row/col 是 1-based 索引 (1 到 N)
			const int maxR = params.sai.rows;
			const int maxC = params.sai.cols;

			while (params.sai.isPlaying) {
				// --- TODO 开始: 计算下一帧坐标 ---
				int r = params.sai.row;
				int c = params.sai.col;

				// 1. 上边缘 (Row=1): 向右走
				if (r == 1 && c < maxC) {
					c++;
				}
				// 2. 右边缘 (Col=Max): 向下走
				else if (c == maxC && r < maxR) {
					r++;
				}
				// 3. 下边缘 (Row=Max): 向左走
				else if (r == maxR && c > 1) {
					c--;
				}
				// 4. 左边缘 (Col=1): 向上走
				else if (c == 1 && r > 1) {
					r--;
				}
				// 5. 如果当前点不在边缘上（比如一开始就在中间），或者 1x1
				// 的情况
				else {
					// 强制归位到左上角，开始循环
					r = 1;
					c = 1;
					// 如果网格大于 1x1，下一步移动到 (1,2)
					if (maxC > 1)
						c = 2;
					else if (maxR > 1)
						r = 2;
				}

				// 更新状态
				params.sai.row = r;
				params.sai.col = c;
				// --- TODO 结束 ---

				// 发送信号更新界面
				// 注意：getSAI 使用 0-based 索引，所以这里减 1
				emit imageReady(ImageType::Center,
								cvMatToQImage(lf->getSAI(params.sai.row - 1,
														 params.sai.col - 1)));
				emit paramsChanged();
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		},
		"SAI played");
}

void LFControl::refocus() {
	runAsync(
		[this] {
			ref->setLF(lf);
			auto img =
				ref->refocusByShift(params.refocus.shift, params.refocus.crop);
			// auto img =
			// 	ref->refocusByAlpha(params.refocus.alpha, params.refocus.crop);
			emit imageReady(ImageType::Refocus, cvMatToQImage(img));
			emit paramsChanged();
		},
		"Refocus");
}

void LFControl::processAllInFocus() {
	runAsync(
		[this] {
			ref->setLF(lf);
			// auto img = ref->generateAllInFocus(-2.0f, 0.0f, 0.05f);
			auto img =
				ref->refocusByShift(params.refocus.shift, params.refocus.crop);
			emit imageReady(ImageType::Refocus, cvMatToQImage(img));
			emit paramsChanged();
		},
		"processAllInFocus");
}

void LFControl::upsample() {
	runAsync(
		[this] {
			sr->setType(params.sr.type);
			auto img = sr->upsample(lf->getCenter());
			emit imageReady(ImageType::SR, cvMatToQImage(img));
			emit paramsChanged();
		},
		"Super resolution");
}

void LFControl::depth() {
	runAsync(
		[this] {
			dep->setType(params.de.type);
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
		// 这里必须用到 params.isp.bitDepth
		else if (inMat.depth() == CV_16U || inMat.depth() == CV_16S) {
			int validBits = params.isp.bitDepth;

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
