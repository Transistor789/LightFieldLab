#include "lfcalibrate.h"

#include "centers_extract.h"
#include "centers_sort.h"
#include "hexgrid_fit.h"
#include "utils.h"

#include <fstream>
#include <json.hpp>

using json = nlohmann::json;

LFCalibrate::LFCalibrate() {}
LFCalibrate::LFCalibrate(const cv::Mat &white_img)
	: _white_img(white_img.clone()) {}

std::vector<std::vector<cv::Point2f>> LFCalibrate::run(bool use_cca,
													   bool save) {
	CentroidsExtract ce(_white_img);
	ce.run(use_cca);
	std::vector<cv::Point2f> pts = ce.getPoints();
	auto pitch = ce.getPitch();

	CentroidsSort cs(pts, pitch);
	cs.run();
	std::vector<cv::Point2f> pts_sorted = cs.getPoints();
	std::vector<int> size = cs.getPointsSize();
	bool hex_odd = cs.getHexOdd();

	HexGridFitter hgf(pts_sorted, size, hex_odd);
	hgf.fit();
	auto pts_fitted = hgf.predict();
	_points = pts_fitted;

	if (save) {
		if (use_cca) {
			draw_points(_white_img, pts, "../../data/centers_detected_cca.png",
						1, 0, true);
		} else {
			draw_points(_white_img, pts,
						"../../data/centers_detected_moments.png", 1, 0, true);
		}

		draw_points(_white_img, pts_sorted, "../../data/centers_sorted.png", 1,
					0, true);
		draw_points(_white_img, pts_fitted, "../../data/centers_fitted.png", 1,
					0, true);

		savePoints("../../data/centroids.json");
	}

	return pts_fitted;
}

void LFCalibrate::savePoints(const std::string &filename) {
	json j;
	j["rows"] = static_cast<int>(_points.size());
	j["cols"] = _points.empty() ? 0 : static_cast<int>(_points[0].size());

	std::vector<cv::Point2f> flat_data;
	flat_data.reserve(_points.size()
					  * (_points.empty() ? 0 : _points[0].size()));

	for (const auto &row : _points) {
		flat_data.insert(flat_data.end(), row.begin(), row.end());
	}

	j["data"] = flat_data;

	writeJson(filename, j);
}
