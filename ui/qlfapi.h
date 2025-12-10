#ifndef QLFAPI_H
#define QLFAPI_H

#include "config.h"
#include "lfload.h"
#include "lfrefocus.h"
#include "lfsuperres.h"

#include <QDebug>
#include <QObject>
#include <QThread>
#include <memory>

class QLFLoad : public QObject {
	Q_OBJECT
public:
	explicit QLFLoad(QObject *parent = nullptr)
		: QObject(parent), backend(std::make_unique<LFLoad>()) {}

	std::unique_ptr<LFLoad> backend;

public slots:
	static void printThreadId() {
		std::cout << "LFLoad threadId: " << QThread::currentThreadId()
				  << std::endl;
	}
	void loadSAI(const QString &path, bool isRGB) {
		emit sendLfPtr(backend->read_sai(path.toStdString(), isRGB));
	}

signals:
	void sendLfPtr(const LfPtr &);
	void sendMat(cv::Mat);
};

class QLFRefocus : public QObject {
	Q_OBJECT
public:
	explicit QLFRefocus(QObject *parent = nullptr)
		: QObject(parent), backend(std::make_unique<LFRefocus>()) {}

	std::unique_ptr<LFRefocus> backend;

public slots:
	static void printThreadId() {
		std::cout << "LFRefocus threadId: " << QThread::currentThreadId()
				  << std::endl;
	}
	void refocus(float alpha, int crop) {
		cv::Mat result = backend->refocus(alpha, crop);
		if (result.empty()) {
			qDebug() << "Refocus failed!";
			emit sendMat(cv::Mat());
		}
		emit sendMat(result);
	}
	void onUpdateLF(const LfPtr &ptr) const { backend->update(ptr); }

signals:
	void sendMat(const cv::Mat &);
};

class QLFSuperRes : public QObject {
	Q_OBJECT
public:
	explicit QLFSuperRes(QObject *parent = nullptr)
		: QObject(parent), backend(std::make_unique<LFSuperRes>()) {}

	int type() const { return static_cast<int>(backend->type()); }
	double scale() const { return backend->scale(); }

	std::unique_ptr<LFSuperRes> backend;

public slots:
	static void printThreadId() {
		std::cout << "QLFSuperres threadId: " << QThread::currentThreadId()
				  << std::endl;
	}
	void setType(int index) const {
		backend->setType(static_cast<ModelType>(index));
	}
	void setScale(int index) const { backend->setScale(index); }
	void onUpdateLF(const LfPtr &ptr) const { backend->update(ptr); }
	void loadModel() const { backend->ensureModelLoaded(); }
	void upsample(const cv::Mat &src) {
		cv::Mat result = backend->upsample(src);
		emit finished(result);
	}

signals:
	void finished(const cv::Mat &);
};

#endif // QLFLOAD_H
