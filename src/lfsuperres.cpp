#include "lfsuperres.h"

// LFSuperRes::LFSuperRes(QObject *parent) : QObject(parent) {}
void LFSuperRes::setType(int index) {
	_type = index;
	if (_type >= 4) {
		loadModel();
	}
}

void LFSuperRes::setScale(int index) {
	_scale = 2.0 + static_cast<double>(index);
	if (_type >= 4) {
		loadModel();
	}
}
// void LFSuperRes::setGpu(bool isGpu) { _isGpu = isGpu; }
void LFSuperRes::loadModel() {
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
int LFSuperRes::upsample(const cv::Mat &src, cv::Mat &dst) {
	auto start = std::chrono::high_resolution_clock::now();
	if (_type < 4) {
		int type = _type == LANCZOS ? cv::INTER_LANCZOS4 : _type;
		cv::resize(src, dst, cv::Size(), _scale, _scale, type);
		dst.convertTo(dst, CV_8UC(lf->channels));
	} else if (_type < 7) {
		_sr.upsample(src, dst);
	} else {
		/// TODO: OACC LFDA DistgSSR
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	qDebug() << "Upsample finished! "
			 << "SR_type: " << _type << ", Elapsed time: " << duration.count()
			 << " ms";
	return 0;
}
int LFSuperRes::upsample_single(int row, int col, cv::Mat &dst) {
	auto start = std::chrono::high_resolution_clock::now();
	_input = lf->getSAI(row, col);
	if (_type < 4) {
		int type = _type == LANCZOS ? cv::INTER_LANCZOS4 : _type;
		cv::resize(_input, dst, cv::Size(), _scale, _scale, type);
		dst.convertTo(dst, CV_8UC(lf->channels));
	} else if (_type < 7) {
		_sr.upsample(_input, dst);
	} else {
		/// TODO: OACC LFDA DistgSSR
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	qDebug() << "Upsample finished! "
			 << "SR_type: " << _type << ", Elapsed time: " << duration.count()
			 << " ms";
	return 0;
}
void LFSuperRes::upsample_multiple() {}