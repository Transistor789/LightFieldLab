#ifndef HEXGRID_FIT_H
#define HEXGRID_FIT_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// --- hexgrid_fit.h ---

class HexGridFitter {
public:
	/**
	 * @brief 构造函数：现在接受 X 和 Y 坐标矩阵
	 * @param pts_mats {x_mat, y_mat}，由 CentroidsSort::getPointsAsMats() 提供
	 * @param hex_odd 六边形奇偶偏移标志
	 */
	explicit HexGridFitter(const std::pair<cv::Mat, cv::Mat> &pts_mats, bool hex_odd);

	void fit();
	void fitRobust(float threshold = 2.0f);

	/**
	 * @brief 修改后的快速鲁棒拟合
	 */
	void fitFastRobust(float threshold = 2.0f, int sample_size = 1500);

	/**
	 * @brief 预测逻辑：现在返回 X/Y 矩阵对
	 */
	std::pair<cv::Mat, cv::Mat> predict() const;

	struct GridInfo {
		cv::Point2f origin;
		cv::Point2f vector_vertical;
		cv::Point2f vector_horizontal;
		float pitch_row;
		float pitch_col;
		float rmse;
	};

	GridInfo get_grid_info() const;

	inline int rows() const { return _rows; }
	inline int cols() const { return _cols; }
	inline float get_params_at(int i, int j) const { return _params(i, j); }

private:
	// 存储输入的坐标矩阵
	cv::Mat _x_mat;
	cv::Mat _y_mat;

	int _cols = 0;
	int _rows = 0;
	bool _hex_odd = false;

	Eigen::Matrix<float, 3, 2> _params;
	float _rmse = 0.0f;
	bool _is_fitted = false;

	bool isValidPoint(float x, float y) const;
};

#endif