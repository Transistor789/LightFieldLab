
#ifndef CENTROIDS_EXTRACT_H
#define CENTROIDS_EXTRACT_H

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

class CentroidsExtract {
public:
	enum class Method {
		Contour,	 // 轮廓几何中心 (Geometric Centroid)
		GrayGravity, // 灰度重心法 (Intensity-Weighted Centroid / Center of
					 // Gravity)
		CCA, // 连通域分析 (Connected Components Analysis)
	};
	/**
	 * @brief 构造函数
	 * @param img 输入图像（建议为灰度图）
	 * @param precision 插值精度（用于尺度空间最大值定位）
	 * @param cr 裁剪比例因子（crop ratio）
	 */
	CentroidsExtract(const cv::Mat &img, int precision = 10, int cr = 3);

	/**
	 * @brief 主运行入口
	 * @param use_cca 是否使用连通域分析（否则用轮廓）
	 * @return {检测到的中心点列表, {水平pitch, 垂直pitch}}
	 */
	void run(Method method);
	void run(Method method, int diameter);
	std::vector<cv::Point2f> getPoints() { return _points; }
	std::vector<float> getPitch() { return _pitch; };
	int getEstimatedM() const { return _estimatedM; }

private:
	// === 成员变量 ===
	const cv::Mat _img;
	cv::Mat _cropImg;
	cv::Mat _topImg;
	const int _precision;
	const int _CR;
	int _estimatedM;
	std::vector<cv::Mat> _scaleSpace;
	std::vector<cv::Point2f> _points;
	std::vector<float> _pitch;

	void cropImage();
	void createScaleSpace();
	std::string getMethodString(CentroidsExtract::Method method) const;
	int findScaleMax(bool useRelativeMax);
	std::pair<std::vector<float>, std::vector<float>> interpolateMaxima(
		const std::vector<float> &maxima);

	cv::Mat createGaussKernel(int length, float sigma);
	std::vector<int> findRelativeMaximaIndices(const std::vector<float> &y);
	std::vector<float> computeSignedGradient(const std::vector<float> &x,
											 int precision);
	std::vector<cv::Point2f> detectMlaCenters(const cv::Mat &img, int blockSize,
											  Method method);
	cv::Mat ensureGrayUint8(const cv::Mat &img);
	std::vector<float> estimatePitchXY(const std::vector<cv::Point2f> &points,
									   int K = -1, float tol = 3.0f);
};

#endif