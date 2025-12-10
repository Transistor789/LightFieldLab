#include "centers_extract.h"

// 引入所有必要的头文件
#include <algorithm>
#include <cmath>
#include <numeric>

CentroidsExtract::CentroidsExtract(const cv::Mat &img, int precision, int cr)
	: _img(img), _precision(precision), _CR(cr), _estimatedM(0) {
	if (_img.empty()) {
		throw std::runtime_error("Input image is empty.");
	}
}

void CentroidsExtract::run(bool use_cca) {
	// 1. 尺度空间分析，确定 _estimatedM
	createScaleSpace(); // 内部调用 cropImage()
	findScaleMax(true);

	// 2. 全图检测
	_points = detectMlaCenters(_img, false, _estimatedM, use_cca);

	// 3. 裁剪图检测
	auto crop_centers = detectMlaCenters(_cropImg, false, _estimatedM, use_cca);

	// 4. 计算间距
	_pitch = estimatePitchXY(crop_centers);
}

cv::Mat CentroidsExtract::createGaussKernel(int length, float sigma) {
	if (length % 2 == 0)
		length += 1;
	cv::Mat kernel1D = cv::getGaussianKernel(length, sigma, CV_32F);
	return kernel1D * kernel1D.t();
}

std::vector<int> CentroidsExtract::findRelativeMaximaIndices(
	const std::vector<float> &y) {
	std::vector<int> indices;
	for (size_t i = 1; i < y.size() - 1; ++i) {
		if (y[i] > y[i - 1] && y[i] > y[i + 1]) {
			indices.push_back(static_cast<int>(i));
		}
	}
	return indices;
}

std::vector<float> CentroidsExtract::computeSignedGradient(
	const std::vector<float> &x, int /*precision*/) {
	if (x.size() < 2)
		return {};
	std::vector<float> grad(x.size());
	for (size_t i = 0; i < x.size(); ++i) {
		float g;
		if (i == 0)
			g = x[1] - x[0];
		else if (i == x.size() - 1)
			g = x[i] - x[i - 1];
		else
			g = (x[i + 1] - x[i - 1]) / 2.0f;

		grad[i] = -std::copysign(1.0f, g);
	}
	return grad;
}

cv::Mat CentroidsExtract::ensureGrayUint8(const cv::Mat &img) {
	cv::Mat gray = img;
	if (img.channels() == 3 || img.channels() == 4) {
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	}
	if (gray.depth() != CV_8U) {
		gray.convertTo(gray, CV_8U);
	}
	return gray;
}

std::vector<cv::Point2f> CentroidsExtract::detectMlaCenters(const cv::Mat &img,
															bool useDilate,
															int blockSize,
															bool useCCA) {
	cv::Mat gray = ensureGrayUint8(img);
	if (blockSize % 2 == 0)
		blockSize++;
	blockSize = std::max(blockSize, 3);

	cv::Mat binary;
	cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
						  cv::THRESH_BINARY, blockSize, 1.0);

	if (useDilate) {
		cv::Mat kernel =
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::dilate(binary, binary, kernel);
	}

	std::vector<cv::Point2f> centers;
	int h = gray.rows, w = gray.cols;
	float margin = blockSize * 0.75f;

	if (useCCA) {
		cv::Mat labels, stats, centroids;
		int nLabels = cv::connectedComponentsWithStats(binary, labels, stats,
													   centroids, 8, CV_32S);
		for (int i = 1; i < nLabels; ++i) {
			if (stats.at<int>(i, cv::CC_STAT_AREA) <= 1)
				continue;
			float cx = static_cast<float>(centroids.at<double>(i, 0));
			float cy = static_cast<float>(centroids.at<double>(i, 1));
			if (cx >= margin && cx <= w - margin && cy >= margin
				&& cy <= h - margin) {
				centers.emplace_back(cx, cy);
			}
		}
	} else {
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(binary, contours, cv::RETR_EXTERNAL,
						 cv::CHAIN_APPROX_SIMPLE);
		for (const auto &cnt : contours) {
			cv::Moments m = cv::moments(cnt);
			if (m.m00 > 1e-6) {
				float cx = static_cast<float>(m.m10 / m.m00);
				float cy = static_cast<float>(m.m01 / m.m00);
				if (cx >= margin && cx <= w - margin && cy >= margin
					&& cy <= h - margin) {
					centers.emplace_back(cx, cy);
				}
			}
		}
	}
	return centers;
}

std::vector<float> CentroidsExtract::estimatePitchXY(
	const std::vector<cv::Point2f> &points, int K, float tol) {
	if (points.size() < 2)
		return {0.0f, 0.0f};

	size_t N = points.size();
	cv::Mat Y(N, 1, CV_32F);
	for (size_t i = 0; i < N; ++i) {
		Y.at<float>(static_cast<int>(i), 0) = points[i].y;
	}

	// --- 1. 自动估计 K ---
	if (K <= 0) {
		std::vector<float> yVals(points.size());
		std::transform(points.begin(), points.end(), yVals.begin(),
					   [](const cv::Point2f &p) { return p.y; });
		std::sort(yVals.begin(), yVals.end());

		std::vector<float> diffs;
		std::adjacent_difference(yVals.begin(), yVals.end(),
								 std::back_inserter(diffs));
		if (!diffs.empty())
			diffs.erase(diffs.begin());

		if (!diffs.empty()) {
			double mean =
				std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
			double var = 0.0;
			for (float d : diffs) var += (d - mean) * (d - mean);
			double stdDev = std::sqrt(var / diffs.size());

			int gaps = std::count_if(diffs.begin(), diffs.end(), [&](float d) {
				return d > static_cast<float>(mean + stdDev);
			});
			K = gaps + 1;
		} else {
			K = 1;
		}
		K = std::clamp(K, 1, static_cast<int>(N));
	}

	// --- 2. K-means on Y ---
	cv::Mat labels, centers_mat;
	cv::TermCriteria criteria(
		cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.1);
	cv::kmeans(Y, K, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers_mat);

	// --- 3. 排序聚类中心 ---
	std::vector<std::pair<float, int>> centerIndex;
	for (int i = 0; i < K; ++i) {
		centerIndex.emplace_back(centers_mat.at<float>(i, 0), i);
	}
	std::sort(centerIndex.begin(), centerIndex.end());

	std::unordered_map<int, int> remap;
	std::vector<float> sortedCentersY(K);
	for (int newRow = 0; newRow < K; ++newRow) {
		int oldId = centerIndex[newRow].second;
		remap[oldId] = newRow;
		sortedCentersY[newRow] = centerIndex[newRow].first;
	}

	// --- 4. 计算 Pitch X (行内平均) ---
	std::vector<float> allPitchesX;
	for (int r = 0; r < K; ++r) {
		std::vector<float> xCoords;
		for (size_t i = 0; i < N; ++i) {
			int cluster = labels.at<int>(static_cast<int>(i), 0);
			if (remap[cluster] == r) {
				const auto &pt = points[i];
				if (std::abs(pt.y - sortedCentersY[r]) < tol) {
					xCoords.push_back(pt.x);
				}
			}
		}
		if (xCoords.size() < 2)
			continue;

		std::sort(xCoords.begin(), xCoords.end());
		std::vector<float> diffs;
		std::adjacent_difference(xCoords.begin(), xCoords.end(),
								 std::back_inserter(diffs));
		diffs.erase(diffs.begin());

		float rowPitch =
			std::accumulate(diffs.begin(), diffs.end(), 0.0f) / diffs.size();
		allPitchesX.push_back(rowPitch);
	}

	// --- 5. 计算 Pitch Y (行间平均) ---
	std::vector<float> pitchesYDiff;
	if (K > 1) {
		for (int i = 1; i < K; ++i) {
			pitchesYDiff.push_back(sortedCentersY[i] - sortedCentersY[i - 1]);
		}
	}

	// --- 6. 结果汇总 ---
	float avgPitchX =
		allPitchesX.empty()
			? 0.0f
			: std::accumulate(allPitchesX.begin(), allPitchesX.end(), 0.0f)
				  / allPitchesX.size();
	float avgPitchY =
		pitchesYDiff.empty()
			? 0.0f
			: std::accumulate(pitchesYDiff.begin(), pitchesYDiff.end(), 0.0f)
				  / pitchesYDiff.size();

	// 回退到六边形假设
	if (avgPitchY < 1e-6f && avgPitchX > 1e-6f) {
		avgPitchY = avgPitchX * std::sqrt(3.0f) / 2.0f;
	}

	return {avgPitchX, avgPitchY};
}

// =========================================================================
// 核心流程函数实现
// =========================================================================

void CentroidsExtract::cropImage() {
	int S = _img.rows / 2;
	int halfCrop = S / _CR;
	cv::Rect roi(S - halfCrop, S - halfCrop, 2 * halfCrop - 1,
				 2 * halfCrop - 1);

	if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > _img.cols
		|| roi.y + roi.height > _img.rows) {
		throw std::runtime_error("Crop ROI is out of image bounds.");
	}

	_img(roi).convertTo(_cropImg, CV_32F);

	int kernelSize = _cropImg.rows;
	float sigmaVal = static_cast<float>(S) / _CR / 2.0f;
	cv::Mat gaussKernel = createGaussKernel(kernelSize, sigmaVal);
	_topImg = _cropImg.mul(gaussKernel);
}

void CentroidsExtract::createScaleSpace() {
	_scaleSpace.clear();
	cropImage();

	cv::Mat sigOne = createGaussKernel(9, 1.0f);
	cv::Mat sigSqrt = createGaussKernel(9, static_cast<float>(std::sqrt(2.0)));

	cv::Mat current = _topImg.clone();
	while (current.rows >= 3 && current.cols >= 3) {
		cv::Mat gauss1, gauss2;

		cv::filter2D(current, gauss1, CV_32F, sigOne, cv::Point(-1, -1), 0,
					 cv::BORDER_CONSTANT);
		_scaleSpace.emplace_back(-(gauss1 - current));

		cv::filter2D(gauss1, gauss2, CV_32F, sigSqrt, cv::Point(-1, -1), 0,
					 cv::BORDER_CONSTANT);
		_scaleSpace.emplace_back(-(gauss2 - gauss1));

		cv::resize(gauss2, current, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
	}
}

std::pair<std::vector<float>, std::vector<float>>
CentroidsExtract::interpolateMaxima(const std::vector<float> &maxima) {
	// 简化：直接返回原数据 + 索引（nu = 0,1,2,...）
	std::vector<float> nu(maxima.size());
	std::iota(nu.begin(), nu.end(), 0.0f);

	float maxVal = *std::max_element(maxima.begin(), maxima.end());
	std::vector<float> normalized = maxima;
	if (maxVal > 1e-6f) {
		for (auto &v : normalized) v /= maxVal;
	}
	return {normalized, nu};
}

int CentroidsExtract::findScaleMax(bool useRelativeMax) {
	if (_scaleSpace.size() < 3)
		return 0;

	std::vector<float> maxima;
	for (const auto &mat : _scaleSpace) {
		double minVal, maxVal;
		cv::minMaxLoc(mat, &minVal, &maxVal);
		maxima.push_back(static_cast<float>(maxVal));
	}

	auto [interp, nu] = interpolateMaxima(maxima);

	// 1. 找第一个正梯度 (起始点)
	size_t start = 0;
	auto grad = computeSignedGradient(nu, _precision);
	for (size_t i = 0; i < grad.size(); ++i) {
		if (grad[i] > 0) {
			start = (i > 0) ? i - 1 : 0;
			break;
		}
	}

	// 2. 截取子序列
	std::vector<float> subY(interp.begin() + start, interp.end());
	std::vector<float> subX(nu.begin() + start, nu.end());
	if (subY.empty())
		return 0;

	// 3. 全局最大值 (ArgMax)
	auto maxIt = std::max_element(subY.begin(), subY.end());
	float globalMaxVal = *maxIt;
	size_t globalIdx = std::distance(subY.begin(), maxIt);
	float argGlobal = subX[globalIdx];

	// 4. 第一个相对最大值
	float relMaxVal = 0.0f, argRel = 0.0f;
	auto relIndices = findRelativeMaximaIndices(subY);
	if (!relIndices.empty()) {
		size_t firstRel = relIndices.front();
		relMaxVal = subY[firstRel];
		argRel = subX[firstRel];
	}

	// 5. 选择最终尺度
	float chosenNu = argGlobal;
	if (useRelativeMax && argRel > 0 && argGlobal > 0) {
		if ((globalMaxVal / argGlobal) <= (relMaxVal / argRel)) {
			chosenNu = argRel;
		}
	}

	// 6. 计算 M
	float dogSigma = std::pow(2.0f, chosenNu / 2.0f);
	float lapSigma = dogSigma * 1.18f;
	_estimatedM = static_cast<int>(std::round(lapSigma * 4.0f));
	return _estimatedM;
}