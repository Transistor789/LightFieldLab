#include "lfload.h"

#include "lfdata.h"

LFLoad::LFLoad(QObject* parent) : QObject(parent) {}
void LFLoad::printThreadId() {
	std::cout << "LFLoad threadId: " << QThread::currentThreadId() << std::endl;
}
void LFLoad::load(const QString& path_, bool isRGB) {
	std::string path = path_.toStdString();

	auto start = std::chrono::high_resolution_clock::now();

	// 获取所有文件名
	std::vector<std::string> filenames;
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(entry.path().filename().string());
		}
	}
	std::sort(filenames.begin(), filenames.end());

	// 提前分配空间
	std::vector<cv::Mat> temp(filenames.size());
	std::mutex mtx;

	// 使用 futures 来管理异步任务
	std::vector<std::future<void>> futures;

	// 并行读取图像
	for (size_t i = 0; i < filenames.size(); ++i) {
		futures.push_back(std::async(std::launch::async, [&, i, isRGB, path]() {
			std::string filename = path + "/" + filenames[i];
			cv::Mat img;
			if (isRGB) {
				img = cv::imread(filename, cv::IMREAD_COLOR);
			} else {
				img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
			}

			std::lock_guard<std::mutex> lock(mtx);
			temp[i] = std::move(img); // 写入对应位置
		}));
	}

	for (auto& f : futures) f.wait();

	LightField lf = LightField(temp);
	std::cout << "Loading finished!" << std::endl;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

	emit finished(std::make_shared<LightField>(lf));
}
