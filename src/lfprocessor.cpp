#include "lfprocessor.h"

#include "lfdata.h"
#include "lfload.h"
#include "lfrefocus.h"
#include "lfsuperres.h"

LFProcessor::LFProcessor(QObject* parent) : QObject(parent) {
	memset(threads, 0, sizeof(threads));

	pLoad = initWorker<LFLoad>(LOAD);
	pRefocus = initWorker<LFRefocus>(REFOCUS);
	pSuperres = initWorker<LFSuperres>(SUPERRES);

	connect(pLoad, &LFLoad::finished, this, &LFProcessor::onLFUpdated,
			Qt::QueuedConnection);
	connect(this, &LFProcessor::updateLF_float, pRefocus,
			&LFRefocus::onUpdateLF, Qt::QueuedConnection);
	// connect(
	// 	this, &LFProcessor::updateLF_uchar, pSuperres,
	// 	[sr = pSuperres](const LightFieldPtr& ptr) { sr->lf = ptr; },
	// 	Qt::QueuedConnection);
	connect(this, &LFProcessor::updateLF_float, pSuperres,
			&LFSuperres::onUpdateLF, Qt::QueuedConnection);

	// connect(
	// 	this, &LFProcessor::requestUpsampleWithMat, pSuperres,
	// 	[sr = pSuperres](const cv::Mat& src) { sr->upsample_single(src); },
	// 	Qt::QueuedConnection);
	connect(
		this, &LFProcessor::requestUpsampleWithPos, pSuperres,
		[sr = pSuperres](int row, int col) { sr->upsample_single(row, col); },
		Qt::QueuedConnection);
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
T* LFProcessor::initWorker(WorkerType type) {
	T* worker = new T();
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
void LFProcessor::onLFUpdated(const LightFieldPtr& ptr) {
	if (lf && lf_float) {
		lf.reset();
		lf_float.reset();
	}

	lf = ptr;
	lf_float = std::make_shared<LightField>(*ptr);
	lf_float->toFloat();

	sai_row = (1 + ptr->rows) / 2 - 1;
	sai_col = (1 + ptr->cols) / 2 - 1;
	sai = lf->getSAI(sai_row, sai_col);
	emit updateSAI(sai);
	// emit updateLF_uchar(lf);
	emit updateLF_float(lf_float);
}

void LFProcessor::onGpuSliderValueChanged(int value) {
	isGpu = static_cast<bool>(value);
	lf->toGpu();
	lf_float->toGpu();
	// emit updateLF_uchar(lf);
	emit updateLF_float(lf_float);

	pRefocus->setGpu(isGpu);
	pSuperres->setGpu(isGpu);
}
void LFProcessor::onSRButtonClicked() {
	cv::Mat image;
	if (pSuperres->type() >= LFSuperres::EDSR
		&& pSuperres->type() <= LFSuperres::FSRCNN) {
		image = lf->getSAI(sai_row, sai_col);
	} else {
		image = lf_float->getSAI(sai_row, sai_col);
	}
	qDebug() << "the type of image: " << image.type();
	emit requestUpsampleWithPos(sai_row, sai_col);
}