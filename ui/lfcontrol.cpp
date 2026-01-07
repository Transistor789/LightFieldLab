#include "lfcontrol.h"

#include "colormatcher.h"
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
#include <iostream>
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
	io = std::make_unique<LFIO>();

	params = std::make_unique<LFParams>(&cal->config, &isp->get_config());

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
	params->dynamic.isCapturing = false;
	params->dynamic.isProcessing = false;
	params->dynamic.exit = true;

	m_queueCv.notify_all();
}

void LFControl::setCapturing(bool active) {
	bool wasCapturing = params->dynamic.isCapturing.load();

	// 状态未改变则直接返回
	if (wasCapturing == active)
		return;

	params->dynamic.isCapturing.store(active);

	if (active) {
		LOG_INFO("Capture Started");
	} else {
		LOG_INFO("Capture Paused");
		// 可选：停止采集时，通常不需要唤醒谁，因为 captureTask 是轮询的
	}
}

void LFControl::setProcessing(bool active) {
	bool wasProcessing = params->dynamic.isProcessing.load();

	// 状态未改变则直接返回
	if (wasProcessing == active)
		return;

	params->dynamic.isProcessing.store(active);

	if (active) {
		LOG_INFO("Processing Resumed");
		// 【关键点】！！！
		// 消费者线程可能正卡在 cv.wait() 等待。
		// 如果不 notify，即使 params->dynamic.isProcessing 变成了
		// true，线程也不会醒来检查条件， 只能等到下一帧数据入队时才会被唤醒。
		// 强制唤醒它，让它立即检查 (data_queue && params->dynamic.isProcessing)
		m_queueCv.notify_all();
	} else {
		LOG_INFO("Processing Paused");
	}
}

void LFControl::captureTask() {
	if (!cap)
		return;

	while (!params->dynamic.exit.load()) {
		if (!params->dynamic.isCapturing.load()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		cv::Mat img = cap->getFrame();
		// cv::Mat img;
		LOG_INFO("Capturing ...");

		params->dynamic.capFrameCount++;
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
	while (!params->dynamic.exit) {
		cv::Mat img;
		{
			std::unique_lock<std::mutex> lock(m_queueMtx);

			m_queueCv.wait(lock, [this] {
				return params->dynamic.exit.load()
					   || (!data_queue.empty()
						   && params->dynamic.isProcessing.load());
			});

			LOG_INFO("Processing");
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			if (params->dynamic.exit) {
				return;
			}

			if (data_queue.empty() || !params->dynamic.isProcessing)
				continue;

			img = std::move(data_queue.front());
			data_queue.pop();

			isp->set_lf_img(img);
			isp->preview().resample(true);
			lf = std::make_shared<LFData>(isp->getSAIS());
			params->dynamic.procFrameCount++;
		}
	}
}

void LFControl::readSAI(const QString &path) {
	runAsync(
		[this, path] {
			params->path.sai = path.toStdString();
			// 1. 耗时操作
			std::shared_ptr<LFData> tempLF = LFIO::readSAI(path.toStdString());

			// 2. 加锁赋值
			{
				std::lock_guard<std::mutex> lock(m_dataMtx);
				lf = tempLF;
			}

			emit imageReady(ImageType::Center, cvMatToQImage(lf->getCenter()));

			params->sai.width = lf->width;
			params->sai.height = lf->height;
			params->sai.row = (lf->rows + 1) / 2;
			params->sai.col = (lf->cols + 1) / 2;
			params->sai.rows = lf->rows;
			params->sai.cols = lf->cols;

			emit paramsChanged();
		},
		"Loading sub-aperture images");
}

void LFControl::readLFP(const QString &path) {
	runAsync(
		[this, path] {
			params->path.lfp = path.toStdString();
			if (params->imageType == ImageFileType::Lytro) {
				json j;
				lfraw = io->readLFP(path.toStdString(), &j);
				isp->set_config(j);
				params->image.bayer = isp->get_config().bayer;
				params->image.bitDepth = isp->get_config().bitDepth;

			} else if (params->imageType == ImageFileType::Raw) {
				lfraw = io->readLFP(path.toStdString());
			} else {
				lfraw = io->readStandardImage(path.toStdString());
				isp->set_config(LFIsp::Config());
			}
			params->image.height = lfraw.size().height;
			params->image.width = lfraw.size().width;

			emit imageReady(ImageType::LFP,
							cvMatToQImage(lfraw, params->image.bitDepth));
			emit paramsChanged();
		},
		"Loading light field image");
}

void LFControl::readWhite(const QString &path) {
	runAsync(
		[this, path] {
			params->path.white = path.toStdString();
			if (params->imageType == ImageFileType::Lytro) {
				white = io->readLFP(path.toStdString(), nullptr);
				params->image.bitDepth = 10;
				params->image.bayer = BayerPattern::GRBG;
				cal->initConfigLytro2();

			} else if (params->imageType == ImageFileType::Raw) {
				white = io->readLFP(path.toStdString(), nullptr);
			} else {
				white = io->readStandardImage(path.toStdString());
			}
			isp->set_white_img(white);
			params->image.height = white.size().height;
			params->image.width = white.size().width;
			emit imageReady(ImageType::White,
							cvMatToQImage(white, params->image.bitDepth));
			emit paramsChanged();
		},
		"Loading white image");
}

void LFControl::readExtractLUT(const QString &path) {
	runAsync(
		[this, path] {
			params->path.extractLUT = path.toStdString();
			LFIO::loadLookUpTables(path.toStdString(), isp->maps.extract,
								   params->calibrate.views);
			params->sai.cols = params->sai.rows = params->calibrate.views;

			emit paramsChanged();
		},
		"Load extract-lut");
}

void LFControl::readDehexLUT(const QString &path) {
	runAsync(
		[this, path] {
			params->path.dehexLUT = path.toStdString();
			int _;
			LFIO::loadLookUpTables(path.toStdString(), isp->maps.dehex, _);
			emit paramsChanged();
		},
		"Load dehex-lut");
}

void LFControl::calibrate() {
	runAsync(
		[this] {
			cal->setImage(white);
			auto points = cal->run();
			cv::Mat draw =
				draw_points(white, points, "", 1, cv::Scalar(0), false);
			params->sai.cols = params->sai.rows = params->calibrate.views;
			emit imageReady(ImageType::White,
							cvMatToQImage(draw, params->image.bitDepth));
			emit paramsChanged();
		},
		"Calibrating");
}

void LFControl::genLUT() {
	runAsync(
		[this] {
			isp->maps.extract =
				cal->computeExtractMaps(params->calibrate.views);
			isp->maps.dehex = cal->computeDehexMaps();
			if (params->calibrate.saveLUT) {
				io->saveLookUpTables(
					std::format("data/calibration/lut_extract_{}.bin",
								params->calibrate.views),
					isp->maps.extract, params->calibrate.views);
				io->saveLookUpTables("data/calibration/lut_dehex.bin",
									 isp->maps.dehex, 1);
			}
			emit paramsChanged();
		},
		"Calibrating");
}

void LFControl::detectCamera() {
	runAsync(
		[this] {
			params->dynamic.cameraID = cap->getAvailableCameras(3);
			emit paramsChanged();
		},
		"Detecting camera");
}

void LFControl::process() {
	runAsync(
		[this] {
			// 1. [检查] 源数据是否为空
			if (lfraw.empty()) {
				LOG_WARN("[ISP] Cancelled: Source image (lfraw) is empty.");
				return;
			}

			// 2. [检查] 白板图像 (全流程处理通常强制需要白板进行 LSC)
			if (white.empty()) {
				LOG_ERROR(
					"[ISP] Failed: White image is missing. Full ISP pipeline "
					"requires calibration data.");
				return;
			}

			// 3. [设置] 设置图像数据
			isp->set_lf_img(lfraw.clone());

			// 4. Raw 域预处理流程
			if (params->isp.enableDPC) {
				isp->dpc_fast();
			}
			if (params->isp.enableBLC) {
				isp->blc_fast();
			}
			if (params->isp.enableLSC && params->isp.enableAWB) {
				isp->lsc_awb_fused_fast();
			} else if (params->isp.enableLSC) {
				isp->lsc_fast();
			} else if (params->isp.enableAWB) {
				isp->awb_fast();
			}

			// 5. [逻辑判断] 检查是否启用去马赛克
			if (!params->isp.enableDemosaic) {
				// 将当前的 Raw 结果发送给界面显示
				emit imageReady(
					ImageType::LFP,
					cvMatToQImage(isp->getResult(), params->image.bitDepth));

				// 记录警告：流程因配置而停止
				LOG_WARN(
					"[ISP] Pipeline stopped early: 'Demosaic' is disabled in "
					"settings.");
				return; // ---> 提前结束
			}

			// 6. RGB 域处理流程
			isp->demosaic();

			if (params->isp.enableCCM) {
				isp->ccm_fast();
			}

			// 发送去马赛克后的结果
			emit imageReady(
				ImageType::LFP,
				cvMatToQImage(isp->getResult(), params->image.bitDepth));

			// 7. [逻辑判断] 检查是否启用宏像素提取
			if (!params->isp.enableExtract) {
				// 记录警告：流程因配置而停止
				LOG_WARN(
					"[ISP] Pipeline stopped early: 'Extract' is disabled in "
					"settings.");
				return; // ---> 提前结束
			}

			// 8. 光场重采样与生成
			isp->resample(params->isp.enableDehex);

			if (params->isp.enableColorEq) {
				ColorMatcher::equalize(isp->getSAIS(),
									   params->isp.colorEqMethod);
			}

			lf = std::make_shared<LFData>(isp->getSAIS());
			params->sai.cols = lf->cols;
			params->sai.rows = lf->rows;
			params->sai.width = lf->width;
			params->sai.height = lf->height;
			params->sai.col = (1 + params->sai.cols) / 2;
			params->sai.row = (1 + params->sai.rows) / 2;
			emit imageReady(
				ImageType::Center,
				cvMatToQImage(lf->getCenter(), params->image.bitDepth));
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
			isp->preview();
			emit imageReady(ImageType::LFP, cvMatToQImage(isp->getResult()));

			// 5. [逻辑判断] 检查是否启用了光场提取
			if (!params->isp.enableExtract) {
				// 仅警告流程被配置截断
				LOG_WARN(
					"[FastPreview] Pipeline stopped early: 'Extract' is "
					"disabled in settings.");
				return; // ---> 提前结束
			}

			// 6. 光场后续
			isp->resample(params->isp.enableDehex);

			if (params->isp.enableColorEq) {
				ColorMatcher::equalize(isp->getSAIS(),
									   params->isp.colorEqMethod);
			}

			lf = std::make_shared<LFData>(isp->getSAIS());
			params->sai.cols = lf->cols;
			params->sai.rows = lf->rows;
			params->sai.width = lf->width;
			params->sai.height = lf->height;
			params->sai.col = (1 + params->sai.cols) / 2;
			params->sai.row = (1 + params->sai.rows) / 2;
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
	if (!params->sai.isPlaying) {
		return;
	}
	runAsync(
		[this] {
			LOG_INFO("Playing...");

			// 缓存一下边界，使代码更简洁
			// 假设 params->sai.row/col 是 1-based 索引 (1 到 N)
			const int maxR = params->sai.rows;
			const int maxC = params->sai.cols;

			while (params->sai.isPlaying) {
				// --- TODO 开始: 计算下一帧坐标 ---
				int r = params->sai.row;
				int c = params->sai.col;

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
				params->sai.row = r;
				params->sai.col = c;
				// --- TODO 结束 ---

				// 发送信号更新界面
				// 注意：getSAI 使用 0-based 索引，所以这里减 1
				emit imageReady(ImageType::Center,
								cvMatToQImage(lf->getSAI(params->sai.row - 1,
														 params->sai.col - 1)));
				emit paramsChanged();
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		},
		"SAI played");
}

void LFControl::refocus() {
	runAsync(
		[this] {
			// 1. 安全检查
			// 注意：lf 是 shared_ptr，先判空指针，再判内部数据
			if (!lf || lf->data.empty()) {
				LOG_ERROR("[Refocus] Cancelled: Light field data is empty!");
				return;
			}

			ref->setLF(lf);

			// 2. 执行重聚焦
			// 结果可能是 CV_16U (保持原位深) 或 CV_32F (平均值)
			auto img = ref->refocusByShift(params->refocus.shift,
										   params->refocus.crop);

			// 3. [新增] 位深归一化处理
			// 如果结果不是 8 位，或者是浮点型，强制转为 8 位以便显示
			if (img.depth() != CV_8U) {
				double alpha = 1.0;
				double maxVal = (1 << params->image.bitDepth) - 1.0;

				// 防御性检查
				if (maxVal < 255.0)
					maxVal = 255.0;

				// 计算缩放系数 (将 0~65535 映射到 0~255)
				alpha = 255.0 / maxVal;

				// 执行转换
				// 如果 img 是浮点型且范围是 0.0-1.0，这里的 alpha 应该改回
				// 255.0 但通常 Refocus 后的浮点值范围依然是 0~MaxVal
				cv::Mat temp;
				img.convertTo(temp, CV_8U, alpha);
				img = temp;
			}

			// 4. 发送结果
			// 因为上面已经转成了 8 位，所以这里 bitDepth 传 8 即可
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
			auto img = ref->refocusByShift(params->refocus.shift,
										   params->refocus.crop);
			emit imageReady(ImageType::Refocus,
							cvMatToQImage(img, params->image.bitDepth));
			emit paramsChanged();
		},
		"processAllInFocus");
}

void LFControl::upsample() {
	runAsync(
		[this] {
			// 0. [安全检查] 确保光场数据存在
			if (!lf || lf->data.empty()) {
				LOG_WARN("[SR] Cancelled: Light field data is empty.");
				return;
			}

			// 1. 传统插值算法 (Nearest, Bilinear, Bicubic)
			// 支持 8-bit 和 16-bit，无需转换
			if (params->sr.method < LFSuperRes::Method::ESPCN) {
				auto img = sr->upsample(lf->getCenter(), params->sr.method);
				// 结果保持原有位深，需传入 bitDepth 进行正确显示归一化
				emit imageReady(ImageType::SR,
								cvMatToQImage(img, params->image.bitDepth));
			}
			// 2. 单图 DNN 算法 (ESPCN, EDSR, FSRCNN, LapSRN)
			// OpenCV DNN 模块通常仅支持 8-bit 输入
			else if (params->sr.method < LFSuperRes::Method::DISTGSSR) {
				cv::Mat src = lf->getCenter();
				cv::Mat src_8u;

				// 如果不是 8 位，则进行缩放转换
				if (params->image.bitDepth != 8 || src.depth() != CV_8U) {
					double maxVal = (1 << params->image.bitDepth) - 1.0;
					// 防御性编程：避免除以0
					if (maxVal < 255.0)
						maxVal = 255.0;

					// 归一化到 0-255
					src.convertTo(src_8u, CV_8U, 255.0 / maxVal);
				} else {
					src_8u = src; // 已经是 8 位，直接引用
				}

				auto img = sr->upsample(src_8u, params->sr.method);

				// DNN 输出通常是 8-bit，直接显示即可
				emit imageReady(ImageType::SR, cvMatToQImage(img, 8));
			}
			// 3. 光场 DNN 算法 (DISTGSSR 等)
			// 需要输入一组 8-bit 的视图
			else {
				std::vector<cv::Mat> views_8u;
				views_8u.reserve(lf->data.size());

				// 计算转换系数
				double alpha = 1.0;
				bool needConv = (params->image.bitDepth != 8);

				if (needConv) {
					double maxVal = (1 << params->image.bitDepth) - 1.0;
					if (maxVal < 255.0)
						maxVal = 255.0;
					alpha = 255.0 / maxVal;
				}

				// 批量转换所有视图
				for (const auto &view : lf->data) {
					if (needConv || view.depth() != CV_8U) {
						cv::Mat temp;
						view.convertTo(temp, CV_8U, alpha);
						views_8u.push_back(temp);
					} else {
						views_8u.push_back(view);
					}
				}

				auto views = sr->upsample(views_8u, params->sr.method);

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
			// 1. [安全检查] 确保光场数据存在
			// 注意：先检查指针 lf 是否有效，再检查数据
			if (!lf || lf->data.empty()) {
				LOG_WARN(
					"[Depth Estimation] Cancelled: Light field data is empty!");
				return;
			}

			// 2. [数据准备] 转换为 8-bit 用于计算
			// 深度估计算法通常对亮度敏感，且大多基于 0-255 范围设计
			std::vector<cv::Mat> process_views;

			// 检查是否需要转换
			if (params->image.bitDepth != 8) {
				// 预分配内存，避免多次 realloc
				process_views.reserve(lf->data.size());

				// 计算归一化系数
				double maxVal = (1 << params->image.bitDepth) - 1.0;
				// 防御性检查
				if (maxVal < 255.0)
					maxVal = 255.0;
				double alpha = 255.0 / maxVal;

				// 批量转换
				for (const auto &view : lf->data) {
					cv::Mat tmp;
					view.convertTo(tmp, CV_8U, alpha);
					process_views.push_back(tmp);
				}
			} else {
				// 如果已经是 8-bit，直接使用（cv::Mat
				// 拷贝只是拷贝头信息，开销很小）
				process_views = lf->data;
			}

			// 3. [执行算法] 使用处理后的 process_views
			auto img = dep->depth(process_views, params->de.method);

			// 4. [结果显示]
			// getGrayVisual/getJetVisual 内部通常已经将结果转为 8-bit
			// 彩色/灰度图 所以 cvMatToQImage 不需要传
			// bitDepth，直接用默认处理即可
			if (params->de.color == LFParamsDE::Color::Gray) {
				emit imageReady(ImageType::Depth,
								cvMatToQImage(dep->getGrayVisual()));
			} else if (params->de.color == LFParamsDE::Color::Jet) {
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
			if (params->de.color == LFParamsDE::Color::Gray) {
				emit imageReady(ImageType::Depth,
								cvMatToQImage(dep->getGrayVisual()));
			} else if (params->de.color == LFParamsDE::Color::Jet) {
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
			// params->isp.config.bitDepth;
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
