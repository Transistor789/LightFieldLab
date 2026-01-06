#ifndef COLORMATCHER_H
#define COLORMATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>

class ColorMatcher {
public:
	// 算法类型枚举
	enum class Method { Reinhard, HistMatch };

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
	// --- Reinhard 算法相关 ---
	static void reinhard(cv::Mat &src, const cv::Mat &ref);
	// 优化的内部接口：直接传入预计算好的参考图统计信息
	static void reinhardInternal(cv::Mat &src, const cv::Scalar &ref_mean,
								 const cv::Scalar &ref_std);
	static void computeLabStats(const cv::Mat &src, cv::Scalar &mean,
								cv::Scalar &stddev);

	// --- HistMatch 算法相关 ---
	static void histMatch(cv::Mat &src, const cv::Mat &ref);
	static void histMatchChannel(cv::Mat &src, const cv::Mat &ref);
};

#endif // COLORMATCHER_H