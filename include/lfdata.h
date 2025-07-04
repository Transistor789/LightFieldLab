#ifndef LFDATA_H
#define LFDATA_H

class LightField {
public:
	explicit LightField() = default;
	explicit LightField(const std::vector<cv::Mat>& src) {
		data.reserve(src.size());
		for (const auto& mat : src) {
			data.emplace_back(mat.clone()); // 深拷贝每个 cv::Mat
		}
		setParam();
	}
	LightField(const LightField& src) {
		data.reserve(src.data.size());
		for (const auto& mat : src.data) {
			data.emplace_back(mat.clone()); // 深拷贝每个 cv::Mat
		}
		data_gpu.clear();
		setParam();
	}
	~LightField() {
		if (!data_gpu.empty()) {
			for (int i = 0; i < size; i++) {
				data_gpu[i].release();
			}
		}
	}
	void toFloat() {
		if (type == CV_32FC(channels)) {
			return;
		}
		for (int i = 0; i < size; i++) {
			data[i].convertTo(data[i], CV_32FC(channels));
		}
	}
	void toUchar() {
		if (type == CV_8UC(channels)) {
			return;
		}
		for (int i = 0; i < size; i++) {
			cv::Mat temp;
			data[i].convertTo(data[i], CV_8UC(channels));
		}
	}
	cv::Mat getSAI(int row, int col) const { return data[row * rows + col]; }
	cv::Mat getCenter() const { return data[(1 + size) / 2 - 1]; }
	bool empty() const { return data.empty(); }
	void clear() {
		data.clear();
		data_gpu.clear();
	}
	void toGpu() {
		if (data.empty()) {
			qDebug() << "Data in cpu is empty!\n";
			return;
		}
		if (!data_gpu.empty()) {
			qDebug() << "Data in gpu is not empty!\n";
			return;
		}

		data_gpu.resize(size); // 必须 resize，不要只 reserve！
		for (int i = 0; i < size; i++) {
			if (data[i].empty()) {
				qDebug() << "Warning: data[" << i << "] is empty!";
				continue;
			}
			data_gpu[i] = data[i].getUMat(cv::ACCESS_READ);
		}
	}
	void setParam() {
		size = data.size();
		rows = static_cast<int>(std::sqrt(data.size()));
		cols = rows;
		height = data[0].rows;
		width = data[0].cols;
		channels = data[0].channels();
		type = data[0].type();
	}

private:
	void validate(const std::vector<cv::Mat>& src) {
		if (src.empty()) {
			throw std::invalid_argument(
				"LightField requires a non-empty vector");
		}
		int size = src.size();
		int rows = static_cast<int>(std::sqrt(size));
		if (rows * rows != size) {
			throw std::invalid_argument(
				"LightField size must be a perfect square");
		}
		if (src[0].empty()) {
			throw std::invalid_argument("Input images must not be empty");
		}
	}
	void setParam(const std::vector<cv::Mat>& src) {
		size = src.size();
		rows = static_cast<int>(std::sqrt(src.size()));
		cols = rows;
		height = src[0].rows;
		width = src[0].cols;
		channels = src[0].channels();
		type = src[0].type();
	}

public:
	int size = 0;
	int rows = 0;
	int cols = 0;
	int height = 0;
	int width = 0;
	int channels = 0;
	int type = 0;

	std::vector<cv::Mat> data;
	std::vector<cv::UMat> data_gpu;
};
using LightFieldPtr = std::shared_ptr<LightField>;

#endif