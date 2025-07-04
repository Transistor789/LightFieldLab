#include "lfsuperres.h"

LFSuperres::LFSuperres(QObject *parent) : QObject(parent) {}
void LFSuperres::setType(int index) {
	_type = static_cast<SR_type>(index);
	if (_type >= 4) {
		loadModel();
	}
}
void LFSuperres::printThreadId() {
	std::cout << "LFSuperres threadId: " << QThread::currentThreadId()
			  << std::endl;
}
void LFSuperres::setScale(int index) {
	_scale = 2.0 + static_cast<double>(index);
	if (_type >= 4) {
		loadModel();
	}
}
void LFSuperres::setGpu(bool isGpu) { _isGpu = isGpu; }
void LFSuperres::loadModel() {
	int scale_int = static_cast<int>(_scale);
	std::string name;
	std::string suffix = "_x" + std::to_string(scale_int) + ".pb";
	switch (static_cast<int>(_type)) {
		case EDSR:
			name = "edsr";
			break;
		case ESPCN:
			name = "espcn";
			break;
		case FSRCNN:
			name = "fsrcnn";
			break;
		default:
			break;
	}
	std::transform(name.begin(), name.end(), name.begin(),
				   [](unsigned char c) { return std::tolower(c); });

	_sr.readModel(_modelPath + name + suffix);
	_sr.setModel(name, scale_int);
}
void LFSuperres::onUpdateLF(const LightFieldPtr &ptr) { lf_float = ptr; }
void LFSuperres::upsample_single(const cv::Mat &src) {
	cv::Mat result;
	auto start = std::chrono::high_resolution_clock::now();
	if (_type < 4) {
		int type = _type == LANCZOS ? cv::INTER_LANCZOS4 : _type;
		cv::resize(src, result, cv::Size(), _scale, _scale, type);
		result.convertTo(result, CV_8UC(lf->channels));
	} else if (_type < 7) {
		_sr.upsample(src, result);
	} else {
		/// TODO: OACC LFDA DistgSSR
	}
	emit finished(result);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	qDebug() << "Upsample finished! "
			 << "SR_type: " << _type << ", Elapsed time: " << duration.count()
			 << " ms";
}
void LFSuperres::upsample_single(int row, int col) {
	auto start = std::chrono::high_resolution_clock::now();

	if (_type < 4) {
		int type = _type == LANCZOS ? cv::INTER_LANCZOS4 : _type;
		if (_isGpu) {
			_input_gpu = lf_float->getSAI(row, col).getUMat(cv::ACCESS_READ);
			cv::resize(_input_gpu, _output_gpu, cv::Size(), _scale, _scale,
					   type);
			_output_gpu.convertTo(_output_gpu, CV_8UC(lf_float->channels));
			_output = _output_gpu.getMat(cv::ACCESS_READ).clone();
		} else {
			_input = lf_float->getSAI(row, col);
			cv::resize(_input, _output, cv::Size(), _scale, _scale, type);
			_output.convertTo(_output, CV_8UC(lf_float->channels));
		}
	} else if (_type < 7) {
		_input = lf_float->getSAI(row, col);
		_sr.upsample(_input, _output);
	} else {
		/// TODO: OACC LFDA DistgSSR
	}

	emit finished(_output);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	qDebug() << "Upsample finished! "
			 << "SR_type: " << _type << ", Elapsed time: " << duration.count()
			 << " ms";
}
void LFSuperres::upsample_multiple() {}