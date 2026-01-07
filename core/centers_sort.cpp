#include "centers_sort.h"

#include <cmath>
#include <limits>

CentroidsSort::CentroidsSort(const std::vector<cv::Point2f> &points,
							 const std::vector<float> &pitch)
	: _pitch_unit(pitch[0] / 2.0f, pitch[1] / 2.0f) {
	if (points.empty()) {
		// 可选：抛出异常或记录错误
		throw std::invalid_argument("Input points cannot be empty.");
	}

	// 1. 计算 top_left 和 bottom_right
	float min_sum = std::numeric_limits<float>::max();
	float max_sum = std::numeric_limits<float>::lowest();
	_top_left = points[0];
	_bottom_right = points[0];

	for (const auto &pt : points) {
		float s = pt.x + pt.y;
		if (s < min_sum) {
			min_sum = s;
			_top_left = pt;
		}
		if (s > max_sum) {
			max_sum = s;
			_bottom_right = pt;
		}
	}

	// 2. 构建 _idx2pt 映射
	_idx2pt.clear();
	for (const auto &pt : points) {
		// 计算浮点索引
		float fx = pt.x / _pitch_unit.x;
		float fy = pt.y / _pitch_unit.y;

		IntIndex idx{fx, fy};

		_idx2pt[idx] = pt;
	}
}

void CentroidsSort::run() {
	search_clock_diag();
	assign_index();

	// --- 输出调试信息 ---
	std::cout << "[CentroidsSort] Results:" << std::endl;

	// 1. 输出 _hex_odd (布尔值)
	std::cout << "  > Hex Odd: " << (_hex_odd ? "true" : "false") << std::endl;

	// 2. 输出 _centroids_list 数量
	std::cout << "  > Centroids Count: " << _centroids_list.size() << std::endl;

	// 3. 输出 _size 尺寸 (遍历 vector<int>)
	std::cout << "  > Grid Size: [";
	for (size_t i = 0; i < _size.size(); ++i) {
		std::cout << _size[i];
		if (i < _size.size() - 1) {
			std::cout << ", ";
		}
	}
	std::cout << "]" << std::endl;
}

std::vector<cv::Point2f> CentroidsSort::neighbors_by_idx(const IntIndex &idx,
														 int radius) const {
	std::vector<cv::Point2f> neighbors;
	neighbors.reserve((2 * radius + 1)
					  * (2 * radius + 1)); // 预分配最大可能数量

	for (int di = -radius; di <= radius; ++di) {
		for (int dj = -radius; dj <= radius; ++dj) {
			IntIndex key{idx.x + di, idx.y + dj};
			auto it = _idx2pt.find(key);
			if (it != _idx2pt.end()) {
				neighbors.push_back(it->second);
			}
		}
	}

	return neighbors;
}

cv::Point2f CentroidsSort::next_horz(const cv::Point2f &pt, bool inv) const {
	// 1. 计算整数索引（truncation，匹配 np.int32）
	IntIndex idx{pt.x / _pitch_unit.x, pt.y / _pitch_unit.y};

	// 2. 获取候选点
	auto candidates = neighbors_by_idx(idx, 3);
	if (candidates.empty()) {
		return cv::Point2f(NAN, NAN);
	}

	const float y_tol = _pitch_unit.y; // 行容差
	cv::Point2f best_pt(NAN, NAN);
	bool found = false;

	if (!inv) {
		// direction = 'right': x > pt.x，选 x 最小者（最靠近）
		float min_x = std::numeric_limits<float>::max();
		for (const auto &p : candidates) {
			if (p.x <= pt.x)
				continue;
			if (std::abs(p.y - pt.y) > y_tol)
				continue;
			if (p.x < min_x) {
				min_x = p.x;
				best_pt = p;
				found = true;
			}
		}
	} else {
		// direction = 'left': x < pt.x，选 x 最大者（最靠近）
		float max_x = std::numeric_limits<float>::lowest();
		for (const auto &p : candidates) {
			if (p.x >= pt.x)
				continue;
			if (std::abs(p.y - pt.y) > y_tol)
				continue;
			if (p.x > max_x) {
				max_x = p.x;
				best_pt = p;
				found = true;
			}
		}
	}

	return found ? best_pt : cv::Point2f(NAN, NAN);
}

cv::Point2f CentroidsSort::next_vert(const cv::Point2f &pt, bool inv,
									 bool hex_odd) const {
	// 1. 计算索引并获取候选点
	IntIndex idx{pt.x / _pitch_unit.x, pt.y / _pitch_unit.y};
	auto candidates = neighbors_by_idx(idx, 3);
	if (candidates.empty()) {
		return cv::Point2f(NAN, NAN);
	}

	// 2. 计算目标 y 坐标（±2 * pitch_unit = ±原始 pitch）
	float target_y =
		inv ? (pt.y - 2.0f * _pitch_unit.y) : (pt.y + 2.0f * _pitch_unit.y);

	// 3. 计算目标 x（用于距离比较）
	float target_x = hex_odd ? (pt.x + _pitch_unit.x) : (pt.x - _pitch_unit.x);

	// 4. 筛选并找最接近 target_x 的点
	cv::Point2f best_pt(NAN, NAN);
	float min_dx = std::numeric_limits<float>::max();
	bool found = false;

	for (const auto &p : candidates) {
		// 检查 y 是否在目标行附近
		if (std::abs(p.y - target_y) > _pitch_unit.y) {
			continue;
		}

		// 检查 x 方向是否符合六边形偏移规则
		if (hex_odd) {
			if (p.x <= pt.x)
				continue; // 必须在右侧
		} else {
			if (p.x >= pt.x)
				continue; // 必须在左侧
		}

		float dx = std::abs(p.x - target_x);
		if (dx < min_dx) {
			min_dx = dx;
			best_pt = p;
			found = true;
		}
	}

	return found ? best_pt : cv::Point2f(NAN, NAN);
}

std::tuple<cv::Point2f, int> CentroidsSort::count_points_horz(
	const cv::Point2f &start, const cv::Point2f &end, bool inv) {
	cv::Point2f current = start;
	int count = 0;

	while (true) {
		// 检查是否到达终点（atol = 1e-4）
		if (cv::norm(current - end) <= 1e-4f) {
			break;
		}

		// 获取下一个水平点
		cv::Point2f next_pt = next_horz(current, inv);

		// 检查是否无效（无下一个点）
		if (std::isnan(next_pt.x)) {
			break;
		}

		// 检查方向是否合法
		if ((!inv && next_pt.x <= current.x) || // right: x 应增大
			(inv && next_pt.x >= current.x)) {	// left:  x 应减小
			break;
		}

		// 更新并计数
		current = next_pt;
		++count;
	}

	return std::make_tuple(current, count);
}

std::tuple<cv::Point2f, int> CentroidsSort::count_points_vert(
	const cv::Point2f &start, const cv::Point2f &end, bool inv, bool hex_odd) {
	cv::Point2f current = start;
	int count = 0;

	while (true) {
		// 检查是否到达终点（atol = 1e-4）
		if (cv::norm(current - end) <= 1e-4f) {
			break;
		}

		// 获取下一个垂直点
		cv::Point2f next_pt = next_vert(current, inv, hex_odd);

		// 检查是否无效
		if (std::isnan(next_pt.x)) {
			break;
		}

		// 检查方向是否合法
		if ((!inv && next_pt.y <= current.y) || // down: y 应增大
			(inv && next_pt.y >= current.y)) {	// up:   y 应减小
			break;
		}

		// 更新状态
		current = next_pt;
		++count;
		hex_odd = !hex_odd; // 下一行奇偶性翻转
	}

	return std::make_tuple(current, count);
}

std::vector<int> CentroidsSort::_search_clockwise(
	std::vector<cv::Point2f> &corners) {
	cv::Point2f t_l, t_r, b_r, b_l;

	// 1. 初始化四角
	if (corners.empty()) {
		t_l = b_l = _top_left;
		t_r = b_r = _bottom_right;
	} else {
		// corners = [t_l, t_r, b_r, b_l]
		t_l = corners[0];
		t_r = corners[1];
		b_r = corners[2];
		b_l = corners[3];
	}

	// 2. 上边：t_l → t_r （right）
	auto [new_t_r, x_t] = count_points_horz(t_l, t_r, false);
	t_r = new_t_r;

	// 3. 右边：t_r → b_r （down）
	cv::Point2f next_down =
		next_vert(t_r, false, true); // direction=down, hex_odd=true
	bool right_hex_odd = !std::isnan(next_down.x); // 存在 → true；否则 false
	auto [new_b_r, y_r] = count_points_vert(t_r, b_r, false, right_hex_odd);
	b_r = new_b_r;

	// 4. 下边：b_r → b_l （left）
	auto [new_b_l, x_b] = count_points_horz(b_r, b_l, true);
	b_l = new_b_l;

	// 5. 左边：b_l → t_l （up）
	cv::Point2f next_up =
		next_vert(b_l, true, false);		   // direction=up, hex_odd=false
	bool left_hex_odd = std::isnan(next_up.x); // 无下一个点 → true；否则 false
	auto [new_t_l, y_l] = count_points_vert(b_l, t_l, true, left_hex_odd);
	t_l = new_t_l;

	// 6. 更新 corners（输出最终四角）
	corners = {t_l, t_r, b_r, b_l};

	// 7. 返回 [min(x_b, x_t), min(y_l, y_r)]
	return {std::min(x_b, x_t), std::min(y_l, y_r)};
}

std::vector<int> CentroidsSort::_search_counter_clockwise(
	std::vector<cv::Point2f> &corners) {
	cv::Point2f t_l, t_r, b_r, b_l;

	// 1. 初始化四角
	if (corners.empty()) {
		t_l = t_r = _top_left;
		b_l = b_r = _bottom_right;
	} else {
		// corners = [t_l, t_r, b_r, b_l]
		t_l = corners[0];
		t_r = corners[1];
		b_r = corners[2];
		b_l = corners[3];
	}

	// 2. 左边：t_l → b_l （down）
	cv::Point2f next_down = next_vert(t_l, false, false); // down, hex_odd=false
	bool left_hex_odd = std::isnan(next_down.x);		  // hex_odd = (no next)
	auto [new_b_l, y_l] = count_points_vert(t_l, b_l, false, left_hex_odd);
	b_l = new_b_l;

	// 3. 下边：b_l → b_r （right）
	auto [new_b_r, x_b] = count_points_horz(b_l, b_r, false);
	b_r = new_b_r;

	// 4. 右边：b_r → t_r （up）
	cv::Point2f next_up = next_vert(b_r, true, true); // up, hex_odd=true
	bool right_hex_odd = !std::isnan(next_up.x);	  // hex_odd = has next
	auto [new_t_r, y_r] = count_points_vert(b_r, t_r, true, right_hex_odd);
	t_r = new_t_r;

	// 5. 上边：t_r → t_l （left）
	auto [new_t_l, x_t] = count_points_horz(t_r, t_l, true);
	t_l = new_t_l;

	// 6. 更新 corners in-place: [t_l, t_r, b_r, b_l]
	corners = {t_l, t_r, b_r, b_l};

	// 7. 返回 [min(x_b, x_t), min(y_l, y_r)]
	return {std::min(x_b, x_t), std::min(y_l, y_r)};
}

std::vector<int> CentroidsSort::_search_diag(std::vector<cv::Point2f> &corners,
											 bool hex_odd) {
	// 输入必须有4个点
	if (corners.size() != 4) {
		return {}; // invalid input → None
	}

	cv::Point2f t_l = corners[0];
	cv::Point2f t_r = corners[1];
	cv::Point2f b_r_orig = corners[2]; // 原始 b_r 用于比较
	cv::Point2f b_l = corners[3];

	// 1. 上边: t_l → t_r (right)
	auto [new_t_r, x_t] = count_points_horz(t_l, t_r, false);
	t_r = new_t_r;

	// 2. 右边: t_r → b_r (down)
	auto [b_r1, y_r] = count_points_vert(t_r, b_r_orig, false, hex_odd);

	// 3. 左边: t_l → b_l (down)
	auto [new_b_l, y_l] = count_points_vert(t_l, b_l, false, hex_odd);
	b_l = new_b_l;

	// 4. 下边: b_l → b_r (right)
	auto [b_r2, x_b] = count_points_horz(b_l, b_r_orig, false);

	// 5. 一致性检查：b_r1 ≈ b_r_orig 且 b_r2 ≈ b_r_orig ?
	const float tol = 1e-4f;
	bool match1 = (cv::norm(b_r1 - b_r_orig) <= tol);
	bool match2 = (cv::norm(b_r2 - b_r_orig) <= tol);

	if (match1 && match2) {
		// 更新 corners in-place: [t_l, t_r, b_r_orig, b_l]
		corners = {t_l, t_r, b_r_orig, b_l};
		return {std::min(x_b, x_t), std::min(y_l, y_r)};
	} else {
		return {}; // 表示 None
	}
}

void CentroidsSort::search_clock_diag() {
	const float tol = 1e-4f;

	// Lambda: compare two corner lists (4 points)
	auto corners_equal = [&](const std::vector<cv::Point2f> &a,
							 const std::vector<cv::Point2f> &b) -> bool {
		if (a.size() != 4 || b.size() != 4)
			return false;
		for (int i = 0; i < 4; ++i) {
			if (cv::norm(a[i] - b[i]) > tol)
				return false;
		}
		return true;
	};

	// Lambda: run search (cw or ccw)
	auto run_search = [&](bool cw)
		-> std::tuple<std::vector<int>, std::vector<cv::Point2f>, bool> {
		std::vector<cv::Point2f> corners; // empty → use internal bounds
		std::vector<int> prev_size;
		std::vector<cv::Point2f> prev_corners;

		// 3 iterations to stabilize
		for (int i = 0; i < 3; ++i) {
			std::vector<int> size;
			if (cw) {
				size = _search_clockwise(corners);
			} else {
				size = _search_counter_clockwise(corners);
			}

			// Check convergence
			if (!prev_size.empty() && prev_size == size
				&& corners_equal(prev_corners, corners)) {
				break;
			}
			prev_size = size;
			prev_corners =
				corners; // corners updated in-place by search functions
		}

		// Try diag with hex_odd = false and true
		auto corners_even = corners;
		auto corners_odd = corners;

		auto size_even = _search_diag(corners_even, false);
		auto size_odd = _search_diag(corners_odd, true);

		struct Candidate {
			std::vector<int> size;
			std::vector<cv::Point2f> corners;
			bool hex_odd;
		};
		std::vector<Candidate> candidates;

		if (!size_even.empty()) {
			candidates.push_back({size_even, corners_even, false});
		}
		if (!size_odd.empty()) {
			candidates.push_back({size_odd, corners_odd, true});
		}

		if (candidates.empty()) {
			return {{}, {}, false}; // invalid
		}

		// Select candidate with max area (cols * rows)
		auto best =
			std::max_element(candidates.begin(), candidates.end(),
							 [](const Candidate &a, const Candidate &b) {
								 int area_a = a.size[0] * a.size[1];
								 int area_b = b.size[0] * b.size[1];
								 return area_a < area_b;
							 });

		return {best->size, best->corners, best->hex_odd};
	};

	// Run both directions
	auto [size_cw, corners_cw, hex_odd_cw] = run_search(true);
	auto [size_ccw, corners_ccw, hex_odd_ccw] = run_search(false);

	// Compute areas (handle empty case)
	auto area = [](const std::vector<int> &s) -> int {
		return s.size() == 2 ? s[0] * s[1] : -1;
	};

	int area_cw = area(size_cw);
	int area_ccw = area(size_ccw);

	// Choose better one
	if (area_cw >= area_ccw && area_cw > 0) {
		_size = size_cw;
		_start = corners_cw[0];
		_hex_odd = hex_odd_cw;
	} else if (area_ccw > 0) {
		_size = size_ccw;
		_start = corners_ccw[0];
		_hex_odd = hex_odd_ccw;
	}
	// If both invalid, leave members unchanged (or handle as needed)
}

void CentroidsSort::assign_index() {
	if (_size.size() != 2 || _size[0] <= 0 || _size[1] <= 0) {
		_centroids_list.clear();
		return;
	}

	int cols = _size[0]; // width (x direction)
	int rows = _size[1]; // height (y direction)

	std::vector<std::vector<cv::Point2f>> centroids_list;
	centroids_list.reserve(rows);
	_centroids_list.reserve(rows * cols);

	cv::Point2f curr_row_start = _start;
	bool hex_odd = _hex_odd;

	for (int t = 0; t < rows; ++t) {
		cv::Point2f curr = curr_row_start;
		// std::vector<cv::Point2f> row_list;
		// _centroids_list.reserve(cols);

		// First point in row: store as (y, x) -> Point2f(y, x)
		_centroids_list.emplace_back(curr.x, curr.y);

		// Add remaining cols - 1 points to the right
		for (int s = 0; s < cols - 1; ++s) {
			cv::Point2f nxt = next_horz(curr, false); // 'right' -> inv=false
			if (std::isnan(nxt.x))
				break;									// safety
			_centroids_list.emplace_back(nxt.x, nxt.y); // (y, x)
			curr = nxt;
		}

		// centroids_list.push_back(std::move(row_list));

		// Move to next row
		cv::Point2f next_row_start =
			next_vert(curr_row_start, false, hex_odd); // 'down' -> inv=false
		if (std::isnan(next_row_start.x)) {
			// 如果提前断了，就停止（但通常 size 已校验，不应发生）
			break;
		}
		curr_row_start = next_row_start;
		hex_odd = !hex_odd;
	}

	// _centroids_list = std::move(centroids_list);
}