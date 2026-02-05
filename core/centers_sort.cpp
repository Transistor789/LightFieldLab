#include "centers_sort.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <unordered_map>

CentroidsSort::CentroidsSort(const std::vector<cv::Point2f> &points, const std::vector<float> &pitch)
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

std::vector<cv::Point2f> CentroidsSort::neighbors_by_idx(const IntIndex &idx, int radius) const {
	std::vector<cv::Point2f> neighbors;
	neighbors.reserve((2 * radius + 1) * (2 * radius + 1)); // 预分配最大可能数量

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
		return cv::Point2f(-1.0f, -1.0f);
	}

	const float y_tol = _pitch_unit.y; // 行容差
	cv::Point2f best_pt(-1.0f, -1.0f);
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

	return found ? best_pt : cv::Point2f(-1.0f, -1.0f);
}

cv::Point2f CentroidsSort::next_vert(const cv::Point2f &pt, bool inv, bool hex_odd) const {
	// 1. 计算索引并获取候选点
	IntIndex idx{pt.x / _pitch_unit.x, pt.y / _pitch_unit.y};
	auto candidates = neighbors_by_idx(idx, 3);
	if (candidates.empty()) {
		return cv::Point2f(-1.0f, -1.0f);
	}

	// 2. 计算目标 y 坐标（±2 * pitch_unit = ±原始 pitch）
	float target_y = inv ? (pt.y - 2.0f * _pitch_unit.y) : (pt.y + 2.0f * _pitch_unit.y);

	// 3. 计算目标 x（用于距离比较）
	float target_x = hex_odd ? (pt.x + _pitch_unit.x) : (pt.x - _pitch_unit.x);

	// 4. 筛选并找最接近 target_x 的点
	cv::Point2f best_pt(-1.0f, -1.0f);
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

	return found ? best_pt : cv::Point2f(-1.0f, -1.0f);
}

std::tuple<cv::Point2f, int> CentroidsSort::count_points_horz(const cv::Point2f &start, const cv::Point2f &end,
															  bool inv) {
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
		if (next_pt.x < 0) {
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

std::tuple<cv::Point2f, int> CentroidsSort::count_points_vert(const cv::Point2f &start, const cv::Point2f &end,
															  bool inv, bool hex_odd) {
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
		if (next_pt.x < 0) {
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

std::pair<cv::Mat, cv::Mat> CentroidsSort::getPointsAsMats(int crop) const {
	// 1. 基础有效性检查
	if (_size.size() < 2 || _size[0] <= 0 || _size[1] <= 0 || _centroids_list.empty()) {
		return {cv::Mat(), cv::Mat()};
	}

	int old_cols = _size[0]; // 原始网格宽度
	int old_rows = _size[1]; // 原始网格高度

	// 2. 计算裁剪后的新尺寸
	// crop 表示从上下左右各剔除几层
	int new_cols = old_cols - 2 * crop;
	int new_rows = old_rows - 2 * crop;

	// 如果裁剪过多导致没有剩余点，返回空矩阵
	if (new_cols <= 0 || new_rows <= 0) {
		return {cv::Mat(), cv::Mat()};
	}

	// 3. 创建目标矩阵
	cv::Mat x_mat(new_rows, new_cols, CV_32F);
	cv::Mat y_mat(new_rows, new_cols, CV_32F);

	// 4. 执行带偏移的拷贝
	// 从第 crop 行/列开始，到倒数第 crop 行/列结束
	for (int r = crop; r < old_rows - crop; ++r) {
		// 计算原图行指针及目标行指针提高效率
		float *dst_x = x_mat.ptr<float>(r - crop);
		float *dst_y = y_mat.ptr<float>(r - crop);

		for (int c = crop; c < old_cols - crop; ++c) {
			// 从 row-major 的原始列表中提取点
			const cv::Point2f &pt = _centroids_list[r * old_cols + c];

			// 写入裁剪后的新矩阵位置
			dst_x[c - crop] = pt.x;
			dst_y[c - crop] = pt.y;
		}
	}

	return {x_mat, y_mat};
}

std::vector<int> CentroidsSort::_search_clockwise(std::vector<cv::Point2f> &corners) {
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
	cv::Point2f next_down = next_vert(t_r, false, true); // direction=down, hex_odd=true
	bool right_hex_odd = !std::isnan(next_down.x);		 // 存在 → true；否则 false
	auto [new_b_r, y_r] = count_points_vert(t_r, b_r, false, right_hex_odd);
	b_r = new_b_r;

	// 4. 下边：b_r → b_l （left）
	auto [new_b_l, x_b] = count_points_horz(b_r, b_l, true);
	b_l = new_b_l;

	// 5. 左边：b_l → t_l （up）
	cv::Point2f next_up = next_vert(b_l, true, false); // direction=up, hex_odd=false
	bool left_hex_odd = std::isnan(next_up.x);		   // 无下一个点 → true；否则 false
	auto [new_t_l, y_l] = count_points_vert(b_l, t_l, true, left_hex_odd);
	t_l = new_t_l;

	// 6. 更新 corners（输出最终四角）
	corners = {t_l, t_r, b_r, b_l};

	// 7. 返回 [min(x_b, x_t), min(y_l, y_r)]
	return {std::min(x_b, x_t), std::min(y_l, y_r)};
}

std::vector<int> CentroidsSort::_search_counter_clockwise(std::vector<cv::Point2f> &corners) {
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

std::vector<int> CentroidsSort::_search_diag(std::vector<cv::Point2f> &corners, bool hex_odd) {
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
	auto corners_equal = [&](const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) -> bool {
		if (a.size() != 4 || b.size() != 4)
			return false;
		for (int i = 0; i < 4; ++i) {
			if (cv::norm(a[i] - b[i]) > tol)
				return false;
		}
		return true;
	};

	// Lambda: run search (cw or ccw)
	auto run_search = [&](bool cw) -> std::tuple<std::vector<int>, std::vector<cv::Point2f>, bool> {
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
			if (!prev_size.empty() && prev_size == size && corners_equal(prev_corners, corners)) {
				break;
			}
			prev_size = size;
			prev_corners = corners; // corners updated in-place by search functions
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
		auto best = std::max_element(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
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
	auto area = [](const std::vector<int> &s) -> int { return s.size() == 2 ? s[0] * s[1] : -1; };

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
		cv::Point2f next_row_start = next_vert(curr_row_start, false, hex_odd); // 'down' -> inv=false
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

void CentroidsSort::run2(float rotation_deg) {
	// 使用基于全局刚性模型的中心扩散算法
	flood_fill_from_center(rotation_deg);

	// --- 输出调试信息 ---
	std::cout << "[CentroidsSort::run2] Rigid Model Results:" << std::endl;
	std::cout << "  > Rotation: " << rotation_deg << " deg" << std::endl;
	std::cout << "  > Grid Size: [" << _size[0] << ", " << _size[1] << "]" << std::endl;
	std::cout << "  > Valid Centroids: "
			  << std::count_if(_centroids_list.begin(), _centroids_list.end(),
							   [](const cv::Point2f &p) { return p.x >= 0; })
			  << std::endl;
}

cv::Point2f CentroidsSort::find_nearest_existing(const cv::Point2f &target, float radius) const {
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

void CentroidsSort::flood_fill_from_center(float rotation_deg) {
	// 1. 初始化角度和旋转矩阵
	float rad = rotation_deg * (CV_PI / 180.0f);
	float cos_r = std::cos(rad);
	float sin_r = std::sin(rad);

	// 辅助 Lambda：旋转一个 Point2f 向量
	auto rotate_vec = [&](const cv::Point2f &v) {
		return cv::Point2f(v.x * cos_r - v.y * sin_r, v.x * sin_r + v.y * cos_r);
	};

	// 2. 寻找种子点 (保持原有逻辑)
	float img_cx = 0, img_cy = 0;
	int count = 0;
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

	cv::Point2f full_pitch = _pitch_unit * 2.0f;
	cv::Point2f seed_pt = find_nearest_existing(cv::Point2f(img_cx, img_cy), std::max(full_pitch.x, full_pitch.y));
	if (seed_pt.x < 0 && !_idx2pt.empty())
		seed_pt = _idx2pt.begin()->second;
	if (seed_pt.x < 0)
		return;

	// 3. BFS 状态准备
	std::unordered_map<IntIndex, cv::Point2f> sorted_visited;
	std::deque<IntIndex> queue;
	IntIndex start_node(0, 0);
	sorted_visited[start_node] = seed_pt;
	queue.push_back(start_node);

	int min_u = 0, max_u = 0, min_v = 0, max_v = 0;
	float search_radius = std::min(full_pitch.x, full_pitch.y) * 0.45f;

	while (!queue.empty()) {
		IntIndex curr = queue.front();
		queue.pop_front();
		cv::Point2f curr_pt = sorted_visited[curr];

		// 更新逻辑边界
		min_u = std::min(min_u, curr.x);
		max_u = std::max(max_u, curr.x);
		min_v = std::min(min_v, curr.y);
		max_v = std::max(max_v, curr.y);

		// 4. 定义旋转后的 6 个搜索方向 (对于六边形网格，6方向比4方向更稳健)
		struct Dir {
			int du, dv;
			cv::Point2f local_offset;
		};

		// 处理六边形行错位 (Hex Row-Major)
		// 根据当前逻辑行号 curr.y 确定下/上行的偏移方向
		int row_parity = ((curr.y % 2) + 2) % 2;
		float h_shift = (row_parity == 0) ? 0.5f : -0.5f;

		std::vector<Dir> directions = {
			{1, 0, cv::Point2f(full_pitch.x, 0)},							   // 右
			{-1, 0, cv::Point2f(-full_pitch.x, 0)},							   // 左
			{0, 1, cv::Point2f(h_shift * full_pitch.x, full_pitch.y)},		   // 下
			{-1, 1, cv::Point2f((h_shift - 1) * full_pitch.x, full_pitch.y)},  // 下左(针对偶行) 或 下右(针对奇行)
			{0, -1, cv::Point2f(h_shift * full_pitch.x, -full_pitch.y)},	   // 上
			{-1, -1, cv::Point2f((h_shift - 1) * full_pitch.x, -full_pitch.y)} // 上左(针对偶行)
		};

		// 纠正方向向量：如果 row_parity 为 1，我们需要调整邻居的索引偏移
		if (row_parity == 1) {
			directions[3] = {1, 1, cv::Point2f((h_shift + 1) * full_pitch.x, full_pitch.y)};
			directions[5] = {1, -1, cv::Point2f((h_shift + 1) * full_pitch.x, -full_pitch.y)};
		}

		for (const auto &d : directions) {
			IntIndex neighbor_idx(curr.x + d.du, curr.y + d.dv);
			if (sorted_visited.find(neighbor_idx) == sorted_visited.end()) {
				// 【关键修复】将预测向量旋转后再累加
				cv::Point2f rotated_offset = rotate_vec(d.local_offset);
				cv::Point2f pred_pt = curr_pt + rotated_offset;

				cv::Point2f found_pt = find_nearest_existing(pred_pt, search_radius);
				if (found_pt.x >= 0) {
					sorted_visited[neighbor_idx] = found_pt;
					queue.push_back(neighbor_idx);
				}
			}
		}
	}

	// 5. 将排序后的结果回填至类成员
	int cols = max_u - min_u + 1;
	int rows = max_v - min_v + 1;
	_size = {cols, rows};
	_centroids_list.assign(cols * rows, cv::Point2f(-1, -1));

	for (const auto &pair : sorted_visited) {
		int u = pair.first.x - min_u;
		int v = pair.first.y - min_v;
		_centroids_list[v * cols + u] = pair.second;
	}
}
