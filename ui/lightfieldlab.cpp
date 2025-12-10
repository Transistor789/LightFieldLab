#include "lightfieldlab.h"

#include "config.h"
#include "lfdata.h"

#include <QMetaObject>
#include <memory>

LFProcessor::LFProcessor(QObject *parent) : QObject(parent) {
	memset(threads, 0, sizeof(threads));

	qload = initWorker<QLFLoad>(LOAD);
	qrefocus = initWorker<QLFRefocus>(REFOCUS);
	qsuperres = initWorker<QLFSuperRes>(SUPERRES);

	connect(qload, &QLFLoad::sendLfPtr, this, &LFProcessor::onLFUpdated,
			Qt::QueuedConnection);
	connect(this, &LFProcessor::sendLfPtr, qrefocus, &QLFRefocus::onUpdateLF,
			Qt::QueuedConnection);
	connect(this, &LFProcessor::sendLfPtr, qsuperres, &QLFSuperRes::onUpdateLF,
			Qt::QueuedConnection);

	// connect(
	// 	this, &LFProcessor::requestUpsampleWithMat, pSuperres,
	// 	[sr = pSuperres](const cv::Mat& src) { sr->upsample_single(src); },
	// 	Qt::QueuedConnection);
	// connect(
	// 	this, &LFProcessor::requestUpsampleWithPos, qsuperres,
	// 	[sr = qsuperres](int row, int col) { sr->upsample_single(row, col); },
	// 	Qt::QueuedConnection);
}
LFProcessor::~LFProcessor() {
	for (int i = 0; i < WORKER_COUNT; ++i) {
		if (threads[i]) {
			threads[i]->quit();
			threads[i]->wait();
		}
	}
}
void LFProcessor::printThreadId() {
	std::cout << "LFProcessor threadId: " << QThread::currentThreadId()
			  << std::endl;
}
template <typename T>
T *LFProcessor::initWorker(int type) {
	T *worker = new T();
	threads[type] = new QThread(this);

	worker->moveToThread(threads[type]);

	connect(threads[type], &QThread::started, worker, &T::printThreadId,
			Qt::QueuedConnection);

	// connect(threads[type], &QThread::started, worker, [type]() {
	// 	qDebug() << "Worker" << type
	// 			 << "running in thread:" << QThread::currentThreadId();
	// });

	// 清理逻辑
	connect(threads[type], &QThread::finished, threads[type],
			&QObject::deleteLater);
	connect(threads[type], &QThread::finished, worker, &QObject::deleteLater);

	threads[type]->start();
	return worker;
}
void LFProcessor::onLFUpdated(const LfPtr &ptr) {
	lf = ptr;

	sai_row = (1 + ptr->rows) / 2 - 1;
	sai_col = (1 + ptr->cols) / 2 - 1;
	sai = lf->getSAI(sai_row, sai_col);
	emit updateSAI(sai);
	emit sendLfPtr(lf);
}

void LFProcessor::onGpuSliderValueChanged(int value) {
	isGpu = static_cast<bool>(value);
	emit sendLfPtr(lf);
}
void LFProcessor::onSRButtonClicked() {
	// cv::Mat image;
	// if (pSuperres->type() >= LFEnhancer::EDSR
	// 	&& pSuperres->type() <= LFEnhancer::FSRCNN) {
	// 	image = lf->getSAI(sai_row, sai_col);
	// } else {
	// 	image = lf_float->getSAI(sai_row, sai_col);
	// }
	const cv::Mat image = lf->getSAI(sai_row, sai_col);
	// qDebug() << "the type of image: " << image.type();
	// emit requestUpsampleWithPos(sai_row, sai_col);
	// sai =
	QMetaObject::invokeMethod(qsuperres, &QLFSuperRes::upsample,
							  Qt::QueuedConnection, image);
}