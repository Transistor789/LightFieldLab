#ifndef HEXGRID_FIT_H
#define HEXGRID_FIT_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 六边形网格拟合器（Eigen 实现）
 *
 * 输入一个按行优先顺序排列的点集，通过最小二乘法拟合理想仿射网格模型。
 *
 * 坐标约定：
 * - 输入点以 (y, x) 顺序存储（即 Point2f.x = y, Point2f.y = x）
 * - pts_sorted 按 [row0_col0, row0_col1, ..., row1_col0, ...] 顺序
 * - pts_size = {cols, rows}
 */
class HexGridFitter {
public:
	/**
	 * @brief 构造函数
	 * @param pts_sorted 扁平化的点集，按行优先顺序，每个 Point2f = (y, x)
	 * @param pts_size   {cols, rows}，表示网格的尺寸
	 * @param hex_odd    首行为"奇数行"时为 true（决定下一行偏移方向）
	 */
	HexGridFitter(const std::vector<cv::Point2f> &pts_sorted,
				  const std::vector<int> &pts_size, bool hex_odd);

	/**
	 * @brief 执行最小二乘拟合
	 * 拟合模型：[y, x]^T = A * [row, u_eff, 1]^T
	 * 其中 A 是 2x3 矩阵，求解为 3x2 参数矩阵。
	 */
	void fit();

	/**
	 * @brief 生成理想拟合网格
	 * @return 拟合后的网格，形状为 (rows, cols)，每个点为 (y, x)
	 */
	std::vector<std::vector<cv::Point2f>> predict() const;

	/**
	 * @brief 获取物理网格参数
	 */
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

	// 拟合结果缓存 - 明确指定为 3x2 矩阵
	Eigen::Matrix<float, 3, 2>
		_params; // 3x2 矩阵：[ [a0, b0], [a1, b1], [a2, b2] ]
				 // 模型：y = a0*row + a1*u_eff + a2
				 //      x = b0*row + b1*u_eff + b2
	float _rmse = 0.0f;
	bool _is_fitted = false;

	// 内部辅助函数
	bool isValidPoint(const cv::Point2f &pt) const;
};

#endif