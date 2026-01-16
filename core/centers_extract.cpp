#include "centers_extract.h"

// 引入所有必要的头文件
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

CentroidsExtract::CentroidsExtract(const cv::Mat &img, int precision, int cr)
	: _img(img), _precision(precision), _CR(cr), _estimatedM(0) {
	if (_img.empty()) {
		throw std::runtime_error("Input image is empty.");
	}
}

void CentroidsExtract::run(ExtractMethod method) {
	// 1. 尺度空间分析，确定 _estimatedM
	createScaleSpace(); // 内部调用 cropImage()
	findScaleMax(true);

	// 2. 全图检测
	std::vector<cv::Point2f> crop_centers;
	if (method != ExtractMethod::LOG_NMS) {
		_points = detectMlaCenters(_img, _estimatedM, method);
		crop_centers = detectMlaCenters(_cropImg, _estimatedM, method);
		_pitch = estimatePitchXY(crop_centers);
	} else {
		_points = log_nms(_img, static_cast<double>(_estimatedM));
		// crop_centers = log_nms(_cropImg, static_cast<double>(_estimatedM));
		_pitch =
			estimatePitchFromPoints(_points, static_cast<double>(_estimatedM));
	}

	// 3. 裁剪图检测
	// auto crop_centers = detectMlaCenters(_cropImg, _estimatedM, method);

	// 4. 计算间距

	// --- 输出调试信息 ---
	std::cout << std::format("[CentroidsExtract] ExtractMethod: {}, Results:",
							 getMethodString(method))
			  << std::endl;
	std::cout << "  > Diameter: " << _estimatedM << std::endl;
	std::cout << "  > Points Count: " << _points.size() << std::endl;
	std::cout << "  > Pitch: [" << _pitch[0] << ", " << _pitch[1] << "]"
			  << std::endl;
}

void CentroidsExtract::run(ExtractMethod method, int diameter) {
	std::vector<cv::Point2f> crop_centers;
	if (method != ExtractMethod::LOG_NMS) {
		_points = detectMlaCenters(_img, diameter, method);
		crop_centers = detectMlaCenters(_cropImg, diameter, method);
		_pitch = estimatePitchXY(crop_centers);
	} else {
		_points = log_nms(_img, static_cast<double>(diameter));
		// crop_centers = log_nms(_cropImg, static_cast<double>(diameter));
		_pitch =
			estimatePitchFromPoints(_points, static_cast<double>(diameter));
	}

	// --- 输出调试信息 ---
	std::cout << std::format("[CentroidsExtract] ExtractMethod: {}, Results:",
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

std::vector<cv::Point2f> CentroidsExtract::detectMlaCenters(
	const cv::Mat &img, int blockSize, ExtractMethod method) {
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
	if (method == ExtractMethod::CCA) {
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

		if (method == ExtractMethod::GrayGravity) {
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

		} else { // ExtractMethod::Contour
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

std::string CentroidsExtract::getMethodString(ExtractMethod method) const {
	std::string methodStr;
	if (method == ExtractMethod::Contour) {
		methodStr = "Contour";
	} else if (method == ExtractMethod::GrayGravity) {
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

std::vector<cv::Point2f> CentroidsExtract::log_nms(const cv::Mat &img,
												   double M) {
	if (img.empty())
		return {};

	// -----------------------------------------------------------
	// 1. LoG (Laplacian of Gaussian) 计算
	// -----------------------------------------------------------

	// 计算 Sigma，完全复刻 Python 逻辑: sig = int(M/4)/1.18
	double sigma = static_cast<int>(M / 4.0) / 1.18;

	// 核大小自适应 (通常取 6*sigma，且为奇数)
	int ksize = static_cast<int>(sigma * 6);
	if (ksize % 2 == 0)
		ksize++;
	if (ksize < 3)
		ksize = 3;

	// 转为 32F 进行高精度运算
	cv::Mat img_float;
	img.convertTo(img_float, CV_32F);

	// 高斯模糊 + 拉普拉斯 = LoG
	cv::Mat log_img;
	cv::GaussianBlur(img_float, log_img, cv::Size(ksize, ksize), sigma);
	cv::Laplacian(log_img, log_img, CV_32F,
				  3); // kernel_size=3 for Laplacian itself

	// 符号翻转：Python代码里 mexican_hat 是负卷积，
	// OpenCV 的 Laplacian 对亮点(Blob)通常产生负响应，所以这里取反变正
	log_img = -log_img;

	// -----------------------------------------------------------
	// 2. 高效 NMS (基于形态学膨胀) - 替代手动扫描
	// -----------------------------------------------------------

	// 膨胀操作：让每个像素的值变成它周围 3x3 邻域内的最大值
	cv::Mat dilated;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(log_img, dilated, kernel);

	// 计算阈值 (对应 Python: thresh = np.mean(peak_img))
	cv::Scalar mean_val = cv::mean(log_img);
	float thresh = static_cast<float>(mean_val[0]);

	// -----------------------------------------------------------
	// 3. 生成 Mask 并提取坐标
	// -----------------------------------------------------------

	// Mask 1: 局部最大值 (原值 == 膨胀后的值)
	cv::Mat is_max;
	// 使用 >= 是为了处理 float 精度误差，严格最大通常用 == 配合 bitwise比较
	// 但在浮点 LoG 图中，相等即意味着它就是邻域峰值
	cv::compare(log_img, dilated, is_max, cv::CMP_EQ);

	// Mask 2: 强度阈值 (原值 > 均值)
	cv::Mat is_strong;
	cv::compare(log_img, thresh, is_strong, cv::CMP_GT);

	// Mask 3: 正值约束 (原值 > 0，去除 LoG 的负震荡)
	cv::Mat is_positive;
	cv::compare(log_img, 0, is_positive, cv::CMP_GT);

	// 合并 Mask: Final = Max & Strong & Positive
	cv::Mat final_mask;
	cv::bitwise_and(is_max, is_strong, final_mask);
	cv::bitwise_and(final_mask, is_positive, final_mask);

	// -----------------------------------------------------------
	// 4. 坐标提取与边界过滤
	// -----------------------------------------------------------

	std::vector<cv::Point> locations;
	cv::findNonZero(final_mask, locations);

	std::vector<cv::Point2f> centroids;
	centroids.reserve(locations.size());

	// 边界剔除参数 (对应 Python: r = int(M/2)-1)
	int r = static_cast<int>(M / 2) - 1;
	int h = img.rows;
	int w = img.cols;

	// for (const auto &pt : locations) {
	// 	// 边界检查
	// 	if (pt.x > r && pt.y > r && (w - pt.x) > r && (h - pt.y) > r) {
	// 		centroids.emplace_back(static_cast<float>(pt.x),
	// 							   static_cast<float>(pt.y));
	// 	}
	// }

	// 亚像素优化
	for (const auto &pt : locations) {
		// 边界剔除
		if (pt.x <= r || pt.y <= r || (w - pt.x) <= r || (h - pt.y) <= r)
			continue;

		// 嵌入二次拟合逻辑
		float delta_x = 0.0f;
		float delta_y = 0.0f;

		// 安全检查：确保有 3x3 邻域
		if (pt.x > 0 && pt.y > 0 && pt.x < w - 1 && pt.y < h - 1) {
			const float *ptr_curr = log_img.ptr<float>(pt.y);
			float L = ptr_curr[pt.x - 1];
			float C = ptr_curr[pt.x];
			float R = ptr_curr[pt.x + 1];

			// 分母为 2 * (L + R - 2C)
			// 修正后的公式: delta = (L - R) / (2 * (L + R - 2*C))
			float denom_x = 2.0f * (L + R - 2.0f * C);
			if (std::abs(denom_x) > 1e-5)
				delta_x = (L - R) / denom_x;

			const float *ptr_prev = log_img.ptr<float>(pt.y - 1);
			const float *ptr_next = log_img.ptr<float>(pt.y + 1);
			float T = ptr_prev[pt.x];
			float B = ptr_next[pt.x];

			float denom_y = 2.0f * (T + B - 2.0f * C);
			if (std::abs(denom_y) > 1e-5)
				delta_y = (T - B) / denom_y;
		}

		// 钳制 offset，防止异常点
		if (delta_x > 0.5f)
			delta_x = 0.5f;
		else if (delta_x < -0.5f)
			delta_x = -0.5f;
		if (delta_y > 0.5f)
			delta_y = 0.5f;
		else if (delta_y < -0.5f)
			delta_y = -0.5f;

		centroids.emplace_back(pt.x + delta_x, pt.y + delta_y);
	}

	return centroids;
}

std::vector<float> CentroidsExtract::estimatePitchFromPoints(
	const std::vector<cv::Point2f> &points, double diameter) {
	if (points.size() < 2)
		return {0.0f, 0.0f};

	// 辅助 lambda：计算中位数 (鲁棒性比平均值好，能抗干扰)
	auto get_median = [](std::vector<float> &vals) -> float {
		if (vals.empty())
			return 0.0f;
		size_t n = vals.size() / 2;
		// nth_element 比 sort 快，只需要找到中间那个数
		std::nth_element(vals.begin(), vals.begin() + n, vals.end());
		return vals[n];
	};

	// 阈值设定：用于判断两个点是否属于同一行/列，以及间距是否合理
	// 假设 Pitch 不会小于直径的 0.8 倍，也不会大于 1.5 倍 (六边形排布通常 pitch
	// ≈ diameter)
	float min_dist = static_cast<float>(diameter) * 0.8f;
	float max_dist = static_cast<float>(diameter) * 1.5f;
	float align_thresh =
		static_cast<float>(diameter) * 0.5f; // 用于判定行/列对齐的容差

	// --- 1. 计算 Pitch X ---
	// 策略：按 Y 排序，Y 相近的点即为同一行，计算它们的 X 差值
	std::vector<cv::Point2f> sorted_by_y = points;
	std::sort(sorted_by_y.begin(), sorted_by_y.end(),
			  [](const auto &a, const auto &b) {
				  if (std::abs(a.y - b.y) < 1e-3)
					  return a.x < b.x; // Y 相同时按 X 排
				  return a.y < b.y;
			  });

	std::vector<float> diffs_x;
	diffs_x.reserve(points.size());

	for (size_t i = 0; i < sorted_by_y.size() - 1; ++i) {
		const auto &p1 = sorted_by_y[i];
		const auto &p2 = sorted_by_y[i + 1];

		// 判定是否在同一行 (Y 差异很小)
		if (std::abs(p1.y - p2.y) < align_thresh) {
			float dx = std::abs(p1.x - p2.x);
			// 判定 X 间距是否在合理范围内 (排除噪声点)
			if (dx > min_dist && dx < max_dist) {
				diffs_x.push_back(dx);
			}
		}
	}

	// --- 2. 计算 Pitch Y ---
	// 策略：按 X 排序，X 相近的点即为同一列，计算它们的 Y 差值
	std::vector<cv::Point2f> sorted_by_x = points;
	std::sort(sorted_by_x.begin(), sorted_by_x.end(),
			  [](const auto &a, const auto &b) {
				  if (std::abs(a.x - b.x) < 1e-3)
					  return a.y < b.y; // X 相同时按 Y 排
				  return a.x < b.x;
			  });

	std::vector<float> diffs_y;
	diffs_y.reserve(points.size());

	for (size_t i = 0; i < sorted_by_x.size() - 1; ++i) {
		const auto &p1 = sorted_by_x[i];
		const auto &p2 = sorted_by_x[i + 1];

		// 判定是否在同一列 (X 差异很小)
		if (std::abs(p1.x - p2.x) < align_thresh) {
			float dy = std::abs(p1.y - p2.y);
			// 判定 Y 间距是否在合理范围内
			if (dy > min_dist && dy < max_dist) {
				diffs_y.push_back(dy);
			}
		}
	}

	// 计算中位数
	float pitch_x = get_median(diffs_x);
	float pitch_y = get_median(diffs_y);

	// 兜底：如果没算出来（比如点太少），回退到直径估计
	if (pitch_x == 0.0f)
		pitch_x = static_cast<float>(diameter);
	if (pitch_y == 0.0f)
		pitch_y = static_cast<float>(diameter);

	return {pitch_x, pitch_y};
}