//
// Created by mba-m4 on 25-7-18.
//

#ifndef QLFLOAD_H
#define QLFLOAD_H

#include "lfload.h"
#include "lfrefocus.h"
#include "lfsuperres.h"

#include <QObject>
#include <QThread>
#include <memory>

class QLFLoad : public QObject {
	Q_OBJECT
public:
	explicit QLFLoad(QObject* parent = nullptr)
		: QObject(parent), backend(std::make_unique<LFLoad>()) {}

	std::unique_ptr<LFLoad> backend;

public slots:
	static void printThreadId() {
		std::cout << "LFLoad threadId: " << QThread::currentThreadId()
				  << std::endl;
	}
	void loadSAI(const QString& path, bool isRGB) {
		emit sendLfPtr(backend->loadSAI(path.toStdString(), isRGB));
	}
	void loadRaw(const QString& path, int width, int height, int bitDepth) {
		cv::Mat mat;
		int ret = backend->loadRaw(path.toStdString(), mat, LYTRO_WIDTH,
								   LYTRO_HEIGHT, bitDepth);
		emit sendMat(mat);
	}

signals:
	void sendLfPtr(const LightFieldPtr&);
	void sendMat(cv::Mat);
};

class QLFRefocus : public QObject {
	Q_OBJECT
public:
	explicit QLFRefocus(QObject* parent = nullptr)
		: QObject(parent), backend(std::make_unique<LFRefocus>()) {}

	std::unique_ptr<LFRefocus> backend;

public slots:
	static void printThreadId() {
		std::cout << "LFRefocus threadId: " << QThread::currentThreadId()
				  << std::endl;
	}
	void setGpu(bool isGPU) { backend->setGpu(isGPU); }
	void refocus(float alpha, int crop) {
		cv::Mat result;
		if (backend->refocus(result, alpha, crop)) {
			qDebug() << "Refocus failed!";
			emit sendMat(cv::Mat());
		}
		emit sendMat(result);
	}
	void onUpdateLF(const LightFieldPtr& ptr) const {
		backend->onUpdateLF(ptr);
	}

signals:
	void sendMat(const cv::Mat&);
};

class QLFSuperRes : public QObject {
	Q_OBJECT
public:
	explicit QLFSuperRes(QObject* parent = nullptr)
		: QObject(parent), backend(std::make_unique<LFSuperRes>()) {}

	int type() const { return backend->type(); }
	double scale() const { return backend->scale(); }
	void setGpu(bool isGpu) { backend->setGpu(isGpu); }

	std::unique_ptr<LFSuperRes> backend;

public slots:
	static void printThreadId() {
		std::cout << "QLFSuperres threadId: " << QThread::currentThreadId()
				  << std::endl;
	}
	void setType(int index) const { backend->setType(index); }
	void setScale(int index) const { backend->setScale(index); }
	void onUpdateLF(const LightFieldPtr& ptr) const {
		backend->onUpdateLF(ptr);
	}
	void loadModel() const { backend->loadModel(); }
	void upsample(const cv::Mat& src) {
		cv::Mat result;
		backend->upsample(src, result);
		emit finished(result);
	}

signals:
	void finished(const cv::Mat&);
};

#endif // QLFLOAD_H
