#ifndef COLORMATCHER_H
#define COLORMATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>

class ColorMatcher {
public:
	// 算法类型枚举
	enum class Method { Reinhard, HistMatch, MKL, MVGD, HM_MKL_HM, HM_MVGD_HM };

	/**
	 * @brief [核心接口] 对整个光场视角进行色彩一致性矫正
	 * 自动选择中心视角作为 Reference，并行处理其他所有视角
	 */
	static void equalize(std::vector<cv::Mat> &views, Method method);

	/**
	 * @brief 单张图像处理：将 src 的色彩风格迁移到 ref 的风格
	 */
	static void apply(cv::Mat &src, const cv::Mat &ref, Method method);

private:
	// --- Reinhard 算法 ---
	static void reinhard(cv::Mat &src, const cv::Mat &ref);
	static void reinhardInternal(cv::Mat &src, const cv::Scalar &ref_mean,
								 const cv::Scalar &ref_std);
	static void computeLabStats(const cv::Mat &src, cv::Scalar &mean,
								cv::Scalar &stddev);

	// --- HistMatch 算法 ---
	static void histMatch(cv::Mat &src, const cv::Mat &ref);
	static void histMatchChannel(cv::Mat &src, const cv::Mat &ref);

	// --- MKL & MVGD 算法 (RGB空间) ---
	static void mkl(cv::Mat &src, const cv::Mat &ref);
	static void mvgd(cv::Mat &src, const cv::Mat &ref);

	// --- 线性代数辅助函数 ---
	static void computeMeanCov(const cv::Mat &img, cv::Mat &mean, cv::Mat &cov);
	static cv::Mat sqrtMatrix(const cv::Mat &cov);	  // 计算矩阵平方根
	static cv::Mat invSqrtMatrix(const cv::Mat &cov); // 计算矩阵逆平方根
};

#endif // COLORMATCHER_H