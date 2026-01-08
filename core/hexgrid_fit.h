#ifndef HEXGRID_FIT_H
#define HEXGRID_FIT_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

class HexGridFitter {
public:
	HexGridFitter(const std::vector<cv::Point2f> &pts_sorted,
				  const std::vector<int> &pts_size, bool hex_odd);

	void fit();

	/**
	 * @brief [新增] 鲁棒拟合算法 (RANSAC)
	 * 能自动剔除 (-1,-1) 的空点，并抵抗检测噪点
	 * @param threshold 判定内点的距离阈值（像素），默认 2.0
	 */
	void fitRobust(float threshold = 2.0f);
	void fitFastRobust(float threshold = 2.0f, int sample_size = 1500);

	std::vector<std::vector<cv::Point2f>> predict() const;

	struct GridInfo {
		cv::Point2f origin;			   // (y, x)
		cv::Point2f vector_vertical;   // 每行的 (dy, dx)
		cv::Point2f vector_horizontal; // 每列（含偏移）的 (dy, dx)
		float pitch_row;			   // ||vector_vertical||
		float pitch_col;			   // ||vector_horizontal||
		float rmse;
	};

	GridInfo get_grid_info() const;

	// 访问器
	inline int rows() const { return _rows; }
	inline int cols() const { return _cols; }

private:
	std::vector<cv::Point2f> _pts_sorted;
	int _cols = 0;
	int _rows = 0;
	bool _hex_odd = false;

	Eigen::Matrix<float, 3, 2> _params;
	float _rmse = 0.0f;
	bool _is_fitted = false;

	// 内部辅助函数
	bool isValidPoint(const cv::Point2f &pt) const;
};

#endif