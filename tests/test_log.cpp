#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "lfcalibrate.h"
#include "utils.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> log_nms(const cv::Mat &img, double M) {
	if (img.empty())
		return {};

	// -----------------------------------------------------------
	// 1. LoG (Laplacian of Gaussian) 计算
	// -----------------------------------------------------------

	// 计算 Sigma，完全复刻 Python 逻辑: sig = int(M/4)/1.18
	double sigma = static_cast<int>(M / 4.0) / 1.18;

	// 核大小自适应 (通常取 6*sigma，且为奇数)
	int ksize = static_cast<int>(sigma * 6);
	if (ksize % 2 == 0)
		ksize++;
	if (ksize < 3)
		ksize = 3;

	// 转为 32F 进行高精度运算
	cv::Mat img_float;
	img.convertTo(img_float, CV_32F);

	// 高斯模糊 + 拉普拉斯 = LoG
	cv::Mat log_img;
	cv::GaussianBlur(img_float, log_img, cv::Size(ksize, ksize), sigma);
	cv::Laplacian(log_img, log_img, CV_32F,
				  3); // kernel_size=3 for Laplacian itself

	// 符号翻转：Python代码里 mexican_hat 是负卷积，
	// OpenCV 的 Laplacian 对亮点(Blob)通常产生负响应，所以这里取反变正
	log_img = -log_img;

	// -----------------------------------------------------------
	// 2. 高效 NMS (基于形态学膨胀) - 替代手动扫描
	// -----------------------------------------------------------

	// 膨胀操作：让每个像素的值变成它周围 3x3 邻域内的最大值
	cv::Mat dilated;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(log_img, dilated, kernel);

	// 计算阈值 (对应 Python: thresh = np.mean(peak_img))
	cv::Scalar mean_val = cv::mean(log_img);
	float thresh = static_cast<float>(mean_val[0]);

	// -----------------------------------------------------------
	// 3. 生成 Mask 并提取坐标
	// -----------------------------------------------------------

	// Mask 1: 局部最大值 (原值 == 膨胀后的值)
	cv::Mat is_max;
	// 使用 >= 是为了处理 float 精度误差，严格最大通常用 == 配合 bitwise比较
	// 但在浮点 LoG 图中，相等即意味着它就是邻域峰值
	cv::compare(log_img, dilated, is_max, cv::CMP_EQ);

	// Mask 2: 强度阈值 (原值 > 均值)
	cv::Mat is_strong;
	cv::compare(log_img, thresh, is_strong, cv::CMP_GT);

	// Mask 3: 正值约束 (原值 > 0，去除 LoG 的负震荡)
	cv::Mat is_positive;
	cv::compare(log_img, 0, is_positive, cv::CMP_GT);

	// 合并 Mask: Final = Max & Strong & Positive
	cv::Mat final_mask;
	cv::bitwise_and(is_max, is_strong, final_mask);
	cv::bitwise_and(final_mask, is_positive, final_mask);

	// -----------------------------------------------------------
	// 4. 坐标提取与边界过滤
	// -----------------------------------------------------------

	std::vector<cv::Point> locations;
	cv::findNonZero(final_mask, locations);

	std::vector<cv::Point2f> centroids;
	centroids.reserve(locations.size());

	// 边界剔除参数 (对应 Python: r = int(M/2)-1)
	int r = static_cast<int>(M / 2) - 1;
	int h = img.rows;
	int w = img.cols;

	// for (const auto &pt : locations) {
	// 	// 边界检查
	// 	if (pt.x > r && pt.y > r && (w - pt.x) > r && (h - pt.y) > r) {
	// 		centroids.emplace_back(static_cast<float>(pt.x),
	// 							   static_cast<float>(pt.y));
	// 	}
	// }

	// 亚像素优化
	for (const auto &pt : locations) {
		// 边界剔除
		if (pt.x <= r || pt.y <= r || (w - pt.x) <= r || (h - pt.y) <= r)
			continue;

		// 嵌入二次拟合逻辑
		float delta_x = 0.0f;
		float delta_y = 0.0f;

		// 安全检查：确保有 3x3 邻域
		if (pt.x > 0 && pt.y > 0 && pt.x < w - 1 && pt.y < h - 1) {
			const float *ptr_curr = log_img.ptr<float>(pt.y);
			float L = ptr_curr[pt.x - 1];
			float C = ptr_curr[pt.x];
			float R = ptr_curr[pt.x + 1];

			// 分母为 2 * (L + R - 2C)
			// 修正后的公式: delta = (L - R) / (2 * (L + R - 2*C))
			float denom_x = 2.0f * (L + R - 2.0f * C);
			if (std::abs(denom_x) > 1e-5)
				delta_x = (L - R) / denom_x;

			const float *ptr_prev = log_img.ptr<float>(pt.y - 1);
			const float *ptr_next = log_img.ptr<float>(pt.y + 1);
			float T = ptr_prev[pt.x];
			float B = ptr_next[pt.x];

			float denom_y = 2.0f * (T + B - 2.0f * C);
			if (std::abs(denom_y) > 1e-5)
				delta_y = (T - B) / denom_y;
		}

		// 钳制 offset，防止异常点
		if (delta_x > 0.5f)
			delta_x = 0.5f;
		else if (delta_x < -0.5f)
			delta_x = -0.5f;
		if (delta_y > 0.5f)
			delta_y = 0.5f;
		else if (delta_y < -0.5f)
			delta_y = -0.5f;

		centroids.emplace_back(pt.x + delta_x, pt.y + delta_y);
	}

	return centroids;
}
void test_module() {
	cv::Mat img = cv::imread("../data/white.png", cv::IMREAD_UNCHANGED);
	cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
	cv::imshow("", img);
	cv::waitKey();

	auto points = log_nms(img, 5.0);
	std::cout << points.size() << std::endl;

	// cv::Mat draw = draw_points(img, points, "", 0, 0, false);
	// cv::imshow("", draw);
	// cv::waitKey();
	// cv::imwrite("../data/cali_5x5.png", draw);

	CentroidsSort cs(points, {5, 4.33});
	cs.run2();
	bool _hex_odd = cs.getHexOdd();
	auto points_sorted = cs.getPoints();
	std::cout << points_sorted.size() << std::endl;

	// -------------------------------------------------------------------------
	// 5. 网格拟合 (HexGrid Fit)
	// -------------------------------------------------------------------------
	HexGridFitter hgf(cs.getPointsAsMats(), _hex_odd);

	// 使用快速鲁棒拟合
	hgf.fitFastRobust(2.0f, 1500);

	auto points_fitted = hgf.predict();
	std::cout << points_fitted.first.size() << std::endl;
	cv::Mat draw = draw_points(img, points_fitted, 0, 0);
	cv::imwrite("../data/cali_5x5.png", draw);
	// cv::imshow("", draw);
	// cv::waitKey();
}

void test_integration() {
	cv::Mat img = cv::imread("../data/white.png", cv::IMREAD_UNCHANGED);

	LFCalibrate cali;
	CalibrateConfig config;
	config.ceMethod = ExtractMethod::LOG_NMS;
	config.autoEstimate = false;
	config.diameter = 5;
	config.views = 5;
	config.hexgridfit = true;
	config.orientation = Orientation::VERT;

	cali.run(img, config);
	auto pts = cali.getPoints();

	auto draw = draw_points(img, pts, 0, 0);
	cv::imwrite("../data/cali_5x5.png", draw);
}
int main() {
	// test_module();
	test_integration();

	return 0;
}