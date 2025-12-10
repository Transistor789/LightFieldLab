#ifndef CENTROIDS_SORT_H
#define CENTROIDS_SORT_H

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

// 使用 cv:: 作为坐标点类型

// 定义索引类型 (int_x, int_y)
struct IntIndex {
	int x;
	int y;

	IntIndex(int x_, int y_) : x(x_), y(y_) {}
	IntIndex(float x_, float y_) {
		x = static_cast<int>(std::floor(x_));
		y = static_cast<int>(std::floor(y_));
	}
	// 必须为 std::unordered_map 定义相等运算符
	bool operator==(const IntIndex &other) const {
		return x == other.x && y == other.y;
	}
};

// 为 std::unordered_map 定义哈希函数
namespace std {
template <>
struct hash<IntIndex> {
	size_t operator()(const IntIndex &idx) const {
		// 简单的组合哈希，确保 x 和 y 都能影响结果
		return std::hash<int>()(idx.x) ^ (std::hash<int>()(idx.y) << 1);
	}
};
} // namespace std

class CentroidsSort {
public:
	/**
	 * @brief 构造函数
	 * @param points 检测到的中心点列表 (x, y)
	 * @param pitch 网格间距 (pitch_x, pitch_y)
	 */
	CentroidsSort(const std::vector<cv::Point2f> &points,
				  const std::vector<float> &pitch);

	/**
	 * @brief 主运行入口
	 * @return {排序后的中心点列表, 是否为奇数六边形偏移}
	 */
	void run();

	// === 核心查找方法 ===
	std::vector<cv::Point2f> neighbors_by_idx(const IntIndex &idx,
											  int radius = 3) const;
	// cv::Point2f nearest_point(const cv::Point2f &pt, int radius = 3) const;
	cv::Point2f next_horz(const cv::Point2f &pt, bool inv) const;
	cv::Point2f next_vert(const cv::Point2f &pt, bool inv, bool hex_odd) const;
	std::tuple<cv::Point2f, int> count_points_horz(const cv::Point2f &start,
												   const cv::Point2f &end,
												   bool inv);
	std::tuple<cv::Point2f, int> count_points_vert(const cv::Point2f &start,
												   const cv::Point2f &end,
												   bool inv, bool hex_odd);

	cv::Point2f _top_left;
	cv::Point2f _bottom_right;

	std::vector<int> getPointsSize() const { return _size; }
	bool getHexOdd() const { return _hex_odd; }
	std::vector<cv::Point2f> const getPoints() { return _centroids_list; }

private:
	// === 成员变量 ===
	const cv::Point2f _pitch_unit; // pitch / 2

	// 核心查找结构：整数网格索引到实际坐标点的映射
	std::unordered_map<IntIndex, cv::Point2f> _idx2pt;

	// 结果变量
	std::vector<int> _size; // {W, H}
	cv::Point2f _start;		// Top-Left corner
	bool _hex_odd = false;
	std::vector<cv::Point2f> _centroids_list;

	// === 搜索逻辑 ===
	// 返回 {size, corners}
	std::vector<int> _search_clockwise(std::vector<cv::Point2f> &corners);
	std::vector<int> _search_counter_clockwise(
		std::vector<cv::Point2f> &corners);

	// 返回 {size, corners}
	std::vector<int> _search_diag(std::vector<cv::Point2f> &corners,
								  bool hex_odd);

	void search_clock_diag();
	void assign_index();
};

#endif