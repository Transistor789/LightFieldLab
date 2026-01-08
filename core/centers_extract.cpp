#include "centers_extract.h"

// 引入所有必要的头文件
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

CentroidsExtract::CentroidsExtract(const cv::Mat &img, int precision, int cr)
	: _img(img), _precision(precision), _CR(cr), _estimatedM(0) {
	if (_img.empty()) {
		throw std::runtime_error("Input image is empty.");
	}
}

void CentroidsExtract::run(Method method) {
	// 1. 尺度空间分析，确定 _estimatedM
	createScaleSpace(); // 内部调用 cropImage()
	findScaleMax(true);

	// 2. 全图检测
	_points = detectMlaCenters(_img, _estimatedM, method);

	// 3. 裁剪图检测
	auto crop_centers = detectMlaCenters(_cropImg, _estimatedM, method);

	// 4. 计算间距
	_pitch = estimatePitchXY(crop_centers);

	// --- 输出调试信息 ---
	std::cout << std::format("[CentroidsExtract] Method: {}, Results:",
							 getMethodString(method))
			  << std::endl;
	std::cout << "  > Diameter: " << _estimatedM << std::endl;
	std::cout << "  > Points Count: " << _points.size() << std::endl;
	std::cout << "  > Pitch: [" << _pitch[0] << ", " << _pitch[1] << "]"
			  << std::endl;
}

void CentroidsExtract::run(Method method, int diameter) {
	_points = detectMlaCenters(_img, diameter, method);
	auto crop_centers = detectMlaCenters(_cropImg, diameter, method);
	_pitch = estimatePitchXY(crop_centers);

	// --- 输出调试信息 ---
	std::cout << std::format("[CentroidsExtract] Method: {}, Results:",
							 getMethodString(method))
			  << std::endl;
	std::cout << "  > Diameter: " << _estimatedM << std::endl;
	std::cout << "  > Points Count: " << _points.size() << std::endl;
	std::cout << "  > Pitch: [" << _pitch[0] << ", " << _pitch[1] << "]"
			  << std::endl;
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
															int blockSize,
															Method method) {
	// 1. 预处理：转灰度
	cv::Mat gray = ensureGrayUint8(img);
	int h = gray.rows;
	int w = gray.cols;

	// 2. 预处理：自适应二值化
	if (blockSize % 2 == 0)
		blockSize++;
	blockSize = std::max(blockSize, 3);

	cv::Mat binary;
	cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
						  cv::THRESH_BINARY, blockSize, 1);

	// 3. 预处理：形态学去噪 (腐蚀)
	// 腐蚀可以分离粘连的光斑，对微透镜阵列图像非常重要
	int erodeSize = 2;
	cv::Mat element = cv::getStructuringElement(
		cv::MORPH_ELLIPSE, cv::Size(2 * erodeSize + 1, 2 * erodeSize + 1));
	cv::erode(binary, binary, element);

	// 准备结果容器和边界余量
	std::vector<cv::Point2f> centers;
	float margin = blockSize * 0.75f;

	// =========================================================
	// 分支 1: CCA (连通域分析) - 速度快，适合只有位置需求
	// =========================================================
	if (method == Method::CCA) {
		cv::Mat labels, stats, centroids;
		int nLabels = cv::connectedComponentsWithStats(binary, labels, stats,
													   centroids, 8, CV_32S);
		// 预分配内存
		centers.reserve(nLabels);

		for (int i = 1; i < nLabels; ++i) {
			// 过滤噪点
			if (stats.at<int>(i, cv::CC_STAT_AREA) <= 1)
				continue;

			float cx = static_cast<float>(centroids.at<double>(i, 0));
			float cy = static_cast<float>(centroids.at<double>(i, 1));

			// 边界检查
			if (cx >= margin && cx <= w - margin && cy >= margin
				&& cy <= h - margin) {
				centers.emplace_back(cx, cy);
			}
		}
		return centers;
	}

	// =========================================================
	// 分支 2 & 3: 基于轮廓的处理 (Contour & GrayGravity)
	// =========================================================
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binary, contours, cv::RETR_EXTERNAL,
					 cv::CHAIN_APPROX_SIMPLE);
	centers.reserve(contours.size());

	for (const auto &cnt : contours) {
		double cx = 0.0, cy = 0.0;
		bool valid = false;

		if (method == Method::GrayGravity) {
			// --- 灰度重心法 (精度最高，抗饱和性能好) ---

			// 1. 获取边界框并安全裁剪
			cv::Rect r = cv::boundingRect(cnt);
			r = r & cv::Rect(0, 0, w, h); // 交集操作防止越界
			if (r.area() <= 0)
				continue;

			// 2. 准备掩码和 ROI
			cv::Mat grayROI = gray(r);
			cv::Mat maskROI = cv::Mat::zeros(r.size(), CV_8U);

			// 3. 在局部坐标系绘制掩码
			std::vector<cv::Point> localCnt = cnt;
			for (auto &p : localCnt) {
				p.x -= r.x;
				p.y -= r.y;
			}
			cv::drawContours(maskROI,
							 std::vector<std::vector<cv::Point>>{localCnt}, 0,
							 cv::Scalar(255), cv::FILLED);

			// 4. 应用掩码提取有效像素
			// 这里优化一下：不一定要copyTo，可以直接传 mask 给 moments 吗？
			// cv::moments 不支持 mask 参数，所以必须 copyTo 或者自行计算
			cv::Mat weightedROI;
			grayROI.copyTo(weightedROI, maskROI);

			// 5. 计算灰度矩 (binaryImage=false 表示利用像素值权重)
			cv::Moments m = cv::moments(weightedROI, false);

			if (m.m00 > 1e-6) {
				// 转换回全局坐标
				cx = r.x + m.m10 / m.m00;
				cy = r.y + m.m01 / m.m00;
				valid = true;
			}

		} else { // Method::Contour
			// --- 几何形心法 (基于二值形状) ---

			// 直接对轮廓计算矩 (binaryImage=true 效果，只看形状)
			cv::Moments m = cv::moments(cnt);
			if (m.m00 > 1e-6) {
				cx = m.m10 / m.m00;
				cy = m.m01 / m.m00;
				valid = true;
			}
		}

		// 统一保存结果
		if (valid) {
			if (cx >= margin && cx <= w - margin && cy >= margin
				&& cy <= h - margin) {
				centers.emplace_back(static_cast<float>(cx),
									 static_cast<float>(cy));
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

std::string CentroidsExtract::getMethodString(
	CentroidsExtract::Method method) const {
	std::string methodStr;
	if (method == CentroidsExtract::Method::Contour) {
		methodStr = "Contour";
	} else if (method == CentroidsExtract::Method::GrayGravity) {
		methodStr = "GrayGravity";
	} else {
		methodStr = "CCA";
	}
	return methodStr;
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