#ifndef LFPARAMS_H
#define LFPARAMS_H

#include "lfcalibrate.h"
#include "lfdepth.h"
#include "lfisp.h"
#include "lfsr.h"
#include "utils.h"

#include <atomic>
#include <string>
#include <vector>

struct LFParamsPath {
	std::string lfp, sai, white, extractLUT, dehexLUT;
};

struct LFParamsImage {
	BayerPattern bayer = BayerPattern::NONE;
	int height = 0;
	int width = 0;
	int bitDepth = 8;
};

struct LFParamsSAI {
	int height, width;
	int row = 5;
	int col = 5;
	int rows = 9;
	int cols = 9;
	bool isPlaying = false;
};

struct LFParamsCalibrate {
	LFCalibrate::Config config;
};

struct LFParamsDynamic {
	std::vector<int> cameraID;
	std::atomic<bool> exit = false;
	std::atomic<bool> isCapturing = false;
	std::atomic<bool> isProcessing = false;
	std::atomic<int> capFrameCount{0};
	std::atomic<int> procFrameCount{0};
	bool showLFP = true;
	bool showSAI = true;
	BayerPattern bayer; // TODO
	int bitDepth;
	int width, height;
};

struct LFParamsRefocus {
	int crop = 0;
	float alpha = 1.0f;
	float shift = 0.0f;
};

struct LFParamsSR {
	int scale = 2;
	int patchSize = 196;
	int views = 5;
	SRMethod method = SRMethod::NEAREST;
};

struct LFParamsDE {
	enum class Color { Gray, Jet, Plasma };

	DEMethod method = DEMethod::DistgDisp;
	Color color = Color::Gray;
	int patchSize = 196;
	int views = 5;
};

struct LFParams {
	ImageFileType imageType = ImageFileType::Lytro;
	LFParamsImage image;
	LFParamsPath path;
	LFCalibrate::Config calibrate;
	IspConfig isp;
	LFParamsDynamic dynamic;
	LFParamsSAI sai;
	LFParamsRefocus refocus;
	LFParamsSR sr;
	LFParamsDE de;
};

#endif