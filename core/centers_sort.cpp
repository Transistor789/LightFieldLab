#include "centers_sort.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <unordered_map>

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

// =========================================================================
// 新增: run2() 及其辅助函数 (基于泛洪填充的鲁棒排序算法)
// =========================================================================

void CentroidsSort::run2() {
	// 使用鲁棒的中心扩散算法
	flood_fill_from_center();

	// --- 输出调试信息 ---
	std::cout << "[CentroidsSort::run2] Results (FloodFill):" << std::endl;
	std::cout << "  > Hex Odd: " << (_hex_odd ? "true" : "false") << std::endl;
	std::cout << "  > Centroids Count (Valid): "
			  << std::count_if(_centroids_list.begin(), _centroids_list.end(),
							   [](const cv::Point2f &p) { return p.x >= 0; })
			  << " / " << _centroids_list.size() << std::endl;

	std::cout << "  > Grid Size: [";
	for (size_t i = 0; i < _size.size(); ++i) {
		std::cout << _size[i] << (i < _size.size() - 1 ? ", " : "");
	}
	std::cout << "]" << std::endl;
}

cv::Point2f CentroidsSort::find_nearest_existing(const cv::Point2f &target,
												 float radius) const {
	// 计算目标在空间哈希中的索引
	// _pitch_unit 是构造函数中初始化的 (pitch/2)
	IntIndex center_idx{target.x / _pitch_unit.x, target.y / _pitch_unit.y};

	cv::Point2f best_pt(-1, -1);
	float min_dist_sq = radius * radius;

	// 搜索 3x3 邻域 (基于 pitch_unit 的网格)
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			IntIndex key{center_idx.x + dx, center_idx.y + dy};
			// 复用 _idx2pt (已在构造函数中构建)
			auto it = _idx2pt.find(key);
			if (it != _idx2pt.end()) {
				float dist_sq = cv::norm(target - it->second);
				dist_sq *= dist_sq; // 使用平方距离比较，避免开方
				if (dist_sq < min_dist_sq) {
					min_dist_sq = dist_sq;
					best_pt = it->second;
				}
			}
		}
	}
	return best_pt;
}

void CentroidsSort::flood_fill_from_center() {
	// 1. 寻找图像中心的种子点
	// 简单估算图像中心坐标（通过平均值）
	float img_cx = 0, img_cy = 0;
	int count = 0;
	// 采样部分点估算中心 (避免遍历所有点，虽然遍历也不慢)
	for (const auto &pair : _idx2pt) {
		img_cx += pair.second.x;
		img_cy += pair.second.y;
		if (++count > 2000)
			break;
	}
	if (count > 0) {
		img_cx /= count;
		img_cy /= count;
	}

	// 完整 Pitch (x, y) = _pitch_unit * 2.0
	cv::Point2f full_pitch = _pitch_unit * 2.0f;

	cv::Point2f seed_pt = find_nearest_existing(
		cv::Point2f(img_cx, img_cy), std::max(full_pitch.x, full_pitch.y));

	if (seed_pt.x < 0 && !_idx2pt.empty()) {
		// Fallback: 如果中心没点，随便取一个
		seed_pt = _idx2pt.begin()->second;
	}

	if (seed_pt.x < 0) {
		// 极端情况：无点
		_size = {0, 0};
		_centroids_list.clear();
		return;
	}

	// 2. 初始化 BFS
	// 使用 unordered_map 记录访问过的网格坐标 (u, v) -> 实际坐标 Point2f
	// 这里 (u, v) 是逻辑网格坐标，种子点为 (0, 0)
	std::unordered_map<IntIndex, cv::Point2f> visited;
	std::deque<IntIndex> queue;

	IntIndex start_node(0, 0);
	visited[start_node] = seed_pt;
	queue.push_back(start_node);

	// 记录网格边界 (用于后续构建矩形数组)
	int min_u = 0, max_u = 0;
	int min_v = 0, max_v = 0;

	// 搜索半径容差 (防止跳到隔壁的隔壁)
	float search_radius = std::min(full_pitch.x, full_pitch.y) * 0.45f;

	while (!queue.empty()) {
		IntIndex curr = queue.front();
		queue.pop_front();
		cv::Point2f curr_pt = visited[curr];

		// 更新边界
		if (curr.x < min_u)
			min_u = curr.x;
		if (curr.x > max_u)
			max_u = curr.x;
		if (curr.y < min_v)
			min_v = curr.y;
		if (curr.y > max_v)
			max_v = curr.y;

		// 定义 4 个搜索方向 (右, 左, 下, 上)
		struct Dir {
			int du, dv;
			cv::Point2f offset;
		};

		// 计算当前行的六边形偏移特性
		// 假设 v=0 是偶数行(shift 0)，v=1 是奇数行(shift +0.5 pitch_x)
		// 注意处理负数取模: ((v % 2) + 2) % 2
		int row_parity = ((curr.y % 2) + 2) % 2;
		float shift_down =
			(row_parity == 0) ? 0.5f : -0.5f; // 偶->奇(+0.5), 奇->偶(-0.5)
		float shift_up = (row_parity == 0) ? 0.5f : -0.5f; // 同理

		std::vector<Dir> directions = {
			{1, 0, cv::Point2f(full_pitch.x, 0)},	// Right
			{-1, 0, cv::Point2f(-full_pitch.x, 0)}, // Left
			{0, 1,
			 cv::Point2f(shift_down * full_pitch.x, full_pitch.y)},		 // Down
			{0, -1, cv::Point2f(shift_up * full_pitch.x, -full_pitch.y)} // Up
		};

		for (const auto &d : directions) {
			IntIndex neighbor_idx(curr.x + d.du, curr.y + d.dv);

			// 如果未访问过
			if (visited.find(neighbor_idx) == visited.end()) {
				// 预测位置
				cv::Point2f pred_pt = curr_pt + d.offset;

				// 在实际点集中查找是否存在该预测点附近的点
				cv::Point2f found_pt =
					find_nearest_existing(pred_pt, search_radius);

				if (found_pt.x >= 0) {
					// 找到了有效邻居
					visited[neighbor_idx] = found_pt;
					queue.push_back(neighbor_idx);
				}
			}
		}
	}

	// 3. 构建最终网格 (Flatten to vector)
	int cols = max_u - min_u + 1;
	int rows = max_v - min_v + 1;
	_size = {cols, rows};

	_centroids_list.clear();
	_centroids_list.resize(cols * rows,
						   cv::Point2f(-1, -1)); // 默认填充无效值 (-1, -1)

	for (const auto &pair : visited) {
		int u = pair.first.x - min_u; // 归一化到 0..cols-1
		int v = pair.first.y - min_v; // 归一化到 0..rows-1

		// 存储为 Row-Major 格式: index = v * cols + u
		if (u >= 0 && u < cols && v >= 0 && v < rows) {
			_centroids_list[v * cols + u] = pair.second;
		}
	}

	// 4. 确定奇偶性
	// 我们的坐标系是以 seed (v=0) 为基准。
	// 这里简单约定：min_v (Grid的第0行) 相对于 seed 的奇偶性。
	// row_parity(seed=0) = 0 (Even).
	// row_parity(min_v) = ((min_v % 2) + 2) % 2.
	// 如果 min_v 是偶数，则第0行与seed同相 -> hex_odd = false (假设seed是Even)
	// 如果 min_v 是奇数，则第0行与seed反相 -> hex_odd = true.
	int start_row_parity = ((min_v % 2) + 2) % 2;
	_hex_odd = (start_row_parity != 0);

	// 更新 _start (Top-Left point)
	// 注意：_centroids_list[0] 可能是 (-1,-1)，如果左上角缺损。
	// 但 _start
	// 成员变量通常用于旧算法的逻辑，这里为了兼容性可以赋值第一个有效点
	// 或者直接指向理论左上角。run2 主要产生 _centroids_list。
	// 只要 HexGridFitter 能处理 (-1,-1) 的点即可。
}