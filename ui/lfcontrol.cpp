#include "lfcontrol.h"

#include "config.h"
#include "lfcalibrate.h"
#include "lfdata.h"
#include "lfdepth.h"
#include "lfisp.h"
#include "lfparams.h"
#include "lfsr.h"
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
#include <string>
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
		if (params.dynamic.showLFP) {
			emit imageReady(ImageType::LFP,
							cvMatToQImage(img, params.dynamic.bitDepth));
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
			isp->preview(params.isp).resample(true);
			lf = std::make_shared<LFData>(isp->getSAIS());
			params.dynamic.procFrameCount++;
			if (params.dynamic.showLFP) {
				emit imageReady(ImageType::Center,
								cvMatToQImage(lf->getCenter(), 8));
			}
		}
	}
}

void LFControl::readSAI(const QString &path) {
	runAsync(
		[this, path] {
			params.path.sai = path.toStdString();
			// 1. 耗时操作
			std::shared_ptr<LFData> tempLF = LFIO::ReadSAI(path.toStdString());

			// 2. 加锁赋值
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				lf = tempLF;
			}

			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));

			params.sai.row = (lf->rows + 1) / 2;
			params.sai.col = (lf->cols + 1) / 2;
			params.sai.rows = lf->rows;
			params.sai.cols = lf->cols;

			emit paramsChanged();
		},
		"Loading sub-aperture images");
}

void LFControl::readLFP(const QString &path) {
	runAsync(
		[this, path] {
			params.path.lfp = path.toStdString();
			if (params.imageType == ImageFileType::Lytro) {
				json LfpMeta;
				// 读取 LFP 并获取元数据 j
				lfraw = LFIO::ReadLFP(path.toStdString(), LfpMeta);
				isp->parseJsonToConfig(LfpMeta, params.isp);
				params.image.bayer = params.calibrate.bayer = params.isp.bayer;
				params.image.bitDepth = params.calibrate.bitDepth =
					params.isp.bitDepth;

				// 自动加载白图逻辑 ===
				if (!params.path.white.empty()
					&& std::filesystem::is_directory(params.path.white)) {
					// 传入 LFP 路径(用于提取Key) 和 标定目录
					json WhiteMeta;
					cv::Mat autoWhite = LFIO::ReadWhiteImageAuto(
						path.toStdString(), params.path.white, WhiteMeta);
					if (!autoWhite.empty()) {
						white = autoWhite; // 更新类的白图成员变量
						isp->initConfig(white, params.isp);
						LOG_INFO("Auto-loaded white imagesuccessfully.");
						emit imageReady(
							ImageType::White,
							cvMatToQImage(white, params.image.bitDepth));
					}
				}
			} else if (params.imageType == ImageFileType::Raw) {
			} else {
				lfraw = LFIO::ReadStandardImage(path.toStdString());
			}
			params.image.height = lfraw.size().height;
			params.image.width = lfraw.size().width;

			emit imageReady(ImageType::LFP,
							cvMatToQImage(lfraw, params.image.bitDepth));
			emit paramsChanged();
		},
		"Loading light field image");
}

void LFControl::readWhite(const QString &path) {
	runAsync(
		[this, path] {
			params.path.white = path.toStdString();
			json WhiteMeta;
			if (params.imageType == ImageFileType::Lytro) {
				white =
					LFIO::ReadWhiteImageManual(path.toStdString(), WhiteMeta);
				params.image.bitDepth = 10;
				params.image.bayer = BayerPattern::GRBG;
			} else if (params.imageType == ImageFileType::Raw) {
			} else {
				white = LFIO::ReadStandardImage(path.toStdString());
			}
			isp->initConfig(white, params.isp);
			params.image.height = white.size().height;
			params.image.width = white.size().width;
			emit imageReady(ImageType::White,
							cvMatToQImage(white, params.image.bitDepth));
			emit paramsChanged();
		},
		"Loading white image");
}

void LFControl::readExtractLUT(const QString &path) {
	runAsync(
		[this, path] {
			params.path.extractLUT = path.toStdString();
			LFIO::LoadLookUpTables(path.toStdString(), isp->maps.extract,
								   params.calibrate.views);
			params.sai.cols = params.sai.rows = params.calibrate.views;

			emit paramsChanged();
		},
		"Load extract-lut");
}

void LFControl::readDehexLUT(const QString &path) {
	runAsync(
		[this, path] {
			params.path.dehexLUT = path.toStdString();
			int _;
			LFIO::LoadLookUpTables(path.toStdString(), isp->maps.dehex, _);
			emit paramsChanged();
		},
		"Load dehex-lut");
}

void LFControl::calibrate() {
	runAsync(
		[this] {
			cal->setImage(white);
			cal->run(params.calibrate);
			params.calibrate.diameter = cal->getDiameter();
			if (params.calibrate.saveLUT && !cal->isExtractLutEmpty()
				&& !cal->isDehexLutEmpty()) {
				LFIO::SaveLookUpTables(
					std::format("data/calibration/lut_extract_{}.bin",
								params.calibrate.views),
					cal->getExtractMaps(), params.calibrate.views);
				LFIO::SaveLookUpTables("data/calibration/lut_dehex.bin",
									   cal->getDehexMaps(), 1);
			}
			params.sai.cols = params.sai.rows = params.calibrate.views;
			cv::Mat draw = draw_points(white, cal->getPoints(), "", 1,
									   cv::Scalar(0), false);
			emit imageReady(ImageType::White,
							cvMatToQImage(draw, params.image.bitDepth));
			emit paramsChanged();
		},
		"Calibrating");
}

void LFControl::detectCamera() {
	runAsync(
		[this] {
			params.dynamic.cameraID = cap->getAvailableCameras(3);
			emit paramsChanged();
		},
		"Detecting camera");
}

void LFControl::process() {
	runAsync(
		[this] {
			isp->set_lf_img(lfraw.clone()).process(params.isp);

			if (!params.isp.enableExtract) {
				emit imageReady(
					ImageType::LFP,
					cvMatToQImage(isp->getResult(), params.image.bitDepth));
				return; // ---> 提前结束
			}

			lf = std::make_shared<LFData>(isp->getSAIS());
			params.sai.cols = lf->cols;
			params.sai.rows = lf->rows;
			params.sai.col = (1 + params.sai.cols) / 2;
			params.sai.row = (1 + params.sai.rows) / 2;
			emit imageReady(
				ImageType::LFP,
				cvMatToQImage(isp->getResult(), params.image.bitDepth));
			emit imageReady(
				ImageType::Center,
				cvMatToQImage(lf->getCenter(), params.image.bitDepth));
			emit paramsChanged();
		},
		"ISP");
}

void LFControl::fast_preview() {
	runAsync(
		[this] {
			// 1. [检查] 源图像
			if (lfraw.empty()) {
				LOG_WARN(
					"[FastPreview] Cancelled: Source image (lfraw) is empty.");
				return;
			}

			// 2. [强制检查] 白板图像 (必须存在)
			if (white.empty()) {
				LOG_ERROR(
					"[FastPreview] Failed: White image is missing. It is "
					"REQUIRED for preview.");
				return; // ---> 强制终止，不再往下执行
			}

			// 3. [设置] 只有数据齐全了才设置进 ISP
			isp->set_lf_img(lfraw.clone());

			// 4. [检查] LUT 状态
			if (isp->isLutEmpty()) {
				LOG_ERROR(
					"[FastPreview] Failed: LUT is empty. Please calibrate or "
					"load LUT first.");
				return; // ---> 强制终止
			}

			// --- 快速预览管线 ---
			isp->preview(params.isp);
			emit imageReady(ImageType::LFP, cvMatToQImage(isp->getResult()));

			// 5. [逻辑判断] 检查是否启用了光场提取
			if (!params.isp.enableExtract) {
				// 仅警告流程被配置截断
				LOG_WARN(
					"[FastPreview] Pipeline stopped early: 'Extract' is "
					"disabled in settings.");
				return; // ---> 提前结束
			}

			// 6. 光场后续
			isp->resample(false);
			// isp->resample(params.isp.enableDehex);

			lf = std::make_shared<LFData>(isp->getSAIS());
			params.sai.cols = lf->cols;
			params.sai.rows = lf->rows;
			params.sai.col = (1 + params.sai.cols) / 2;
			params.sai.row = (1 + params.sai.rows) / 2;
			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));
			emit paramsChanged();
		},
		"Fast preview");
}

void LFControl::updateSAI(int row, int col) {
	runAsync(
		[this, row, col] {
			emit imageReady(ImageType::Center,
							cvMatToQImage(lf->getSAI(row - 1, col - 1),
										  params.image.bitDepth));
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
														 params.sai.col - 1),
											  params.image.bitDepth));
				emit paramsChanged();
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		},
		"SAI played");
}

void LFControl::refocus() {
	runAsync(
		[this] {
			// 1. 安全检查: 数据是否存在
			if (!lf || lf->data.empty()) {
				LOG_ERROR("[Refocus] Cancelled: Light field data is empty!");
				return;
			}

			// 2. [新增] 严格检查: 必须是 8-bit 数据
			if (lf->data[0].depth() != CV_8U) {
				LOG_ERROR(
					"[Refocus] Error: Input data is not 8-bit! Current depth: "
					+ std::to_string(lf->data[0].depth()));
				return;
			}

			ref->setLF(lf);

			// 3. 执行重聚焦
			// 由于输入保证是 8-bit，且 Refocus 内部已锁定输出 8-bit
			auto img =
				ref->refocusByShift(params.refocus.shift, params.refocus.crop);

			if (img.empty()) {
				LOG_ERROR("[Refocus] Failed: Result image is empty.");
				return;
			}

			// 4. 发送结果
			// 结果是标准的 8-bit，cvMatToQImage 第二个参数传 8 即可
			emit imageReady(ImageType::Refocus, cvMatToQImage(img, 8));
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
			emit imageReady(ImageType::Refocus,
							cvMatToQImage(img, params.image.bitDepth));
			emit paramsChanged();
		},
		"processAllInFocus");
}

void LFControl::upsample() {
	runAsync(
		[this] {
			// 0. [安全检查] 确保光场数据存在
			if (!lf || lf->data.empty()) {
				LOG_WARN(
					"[Super Resolution] Cancelled: Light field data is empty.");
				return;
			}

			if (lf->data[0].depth() != CV_8U) {
				LOG_ERROR(
					"[Super Resolution] Error: Input data is not 8-bit! "
					"Current depth: "
					+ std::to_string(lf->data[0].depth()));
				return;
			}

			if (params.sr.method < SRMethod::ESPCN) {
				auto img = sr->upsample(lf->getCenter(), params.sr.method);

				// 结果必定是 8-bit，直接显示
				emit imageReady(ImageType::SR, cvMatToQImage(img, 8));
			} else if (params.sr.method < SRMethod::DISTGSSR) {
				auto img = sr->upsample(lf->getCenter(), params.sr.method);

				emit imageReady(ImageType::SR, cvMatToQImage(img, 8));
			} else {
				// 直接传入 lf->data，无需再创建 views_8u 临时副本
				auto views = sr->upsample(lf->data, params.sr.method);

				if (!views.empty()) {
					// 显示中间视点
					emit imageReady(ImageType::SR,
									cvMatToQImage(views[views.size() / 2], 8));
				}
			}
		},
		"Super resolution");
}

void LFControl::depth() {
	runAsync(
		[this] {
			if (!lf || lf->data.empty()) {
				LOG_WARN(
					"[Depth Estimation] Cancelled: Light field data is empty!");
				return;
			}

			if (lf->data[0].depth() != CV_8U) {
				LOG_ERROR(
					"[Depth Estimation] Error: Input data is not 8-bit! "
					"Current depth: "
					+ std::to_string(lf->data[0].depth()));
				return;
			}

			bool success = dep->depth(lf->data, params.de.method);

			if (!success) {
				LOG_ERROR(
					"[Depth Estimation] Failed: Algorithm returned error.");
				return;
			}

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
		},
		"Depth estimation");
}

void LFControl::colorChanged(int index) {
	runAsync(
		[this, index] {
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
		},
		"Depth estimation");
}

QImage LFControl::cvMatToQImage(const cv::Mat &inMat, int bitDepth) {
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
		cv::Mat temp;
		double alpha = 1.0;
		double beta = 0.0;

		// 情况 A: 浮点型 (CV_32F / CV_64F)
		if (inMat.depth() == CV_32F || inMat.depth() == CV_64F) {
			alpha = 255.0;
		}

		// 情况 B: 整数型 (CV_16U / CV_16S)
		else if (inMat.depth() == CV_16U || inMat.depth() == CV_16S) {
			// [修改] 使用传入的参数 bitDepth
			int validBits = bitDepth;

			// 如果调用者没传(为0)或者传错了，给予默认值 16 (最安全的方式)
			// 或者你也可以保留原来的逻辑：if(validBits <= 0) validBits =
			// params.isp.bitDepth;
			if (validBits <= 0)
				validBits = 16;
			if (validBits > 16)
				validBits = 16;

			// 计算最大值 (例如 10bit -> 1023)
			double maxVal = (1 << validBits) - 1.0;

			// [可选优化] 防止过曝：
			// 如果你发现图像依然过曝，是因为AWB增益导致数值溢出，
			// 可以给 maxVal 乘一个系数 (Headroom)，例如: maxVal *= 1.5;

			// 归一化系数
			alpha = 255.0 / maxVal;
		}

		// 执行转换: temp = inMat * alpha + beta -> 转为 8位
		inMat.convertTo(temp, CV_8U, alpha, beta);

		// 如果是单通道转为3通道以便显示
		if (temp.channels() == 1) {
			cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
		}

		// 递归调用自己
		// 此时 temp 已经是 CV_8U，bitDepth 参数不再重要，传入 8 即可
		return cvMatToQImage(temp, 8);
	}
}
