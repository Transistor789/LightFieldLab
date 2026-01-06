#ifndef LFPARAMS_H
#define LFPARAMS_H

#include "colormatcher.h"
#include "lfdepth.h"
#include "lfisp.h"
#include "lfsr.h"

#include <atomic>
#include <string>
#include <vector>

struct LFParamsPath {
	std::string lfp, sai, white, extractLUT, dehexLUT;
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
	int diameter = 0;
	bool useCCA = false;
	bool autoEstimate = false;
	bool saveLUT = false;
	int views = 9;
};

struct LFParamsISP {
	bool enableDPC = true;
	bool enableBLC = true;
	bool enableLSC = true;
	bool enableAWB = true;
	bool enableDemosaic = true;
	bool enableCCM = true;
	bool enableGamma = true;
	bool enableExtract = true;
	bool enableDehex = true;
	bool enableColorEq = true;

	LFIsp::IspConfig config;
	LFIsp::Method::DPC dpcMethod = LFIsp::Method::DPC::Diretional;
	LFIsp::Method::Demosaic demosaicMethod = LFIsp::Method::Demosaic::Bilinear;
	ColorMatcher::Method colorEqMethod = ColorMatcher::Method::Reinhard;
};

struct LFParamsDynamic {
	std::vector<int> cameraID;
	std::atomic<bool> exit = false;
	std::atomic<bool> isCapturing = false;
	std::atomic<bool> isProcessing = false;
	std::atomic<int> capFrameCount{0};
	std::atomic<int> procFrameCount{0};
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
	LFSuperRes::Method method = LFSuperRes::Method::NEAREST;
};

struct LFParamsDE {
	enum class Color { Gray, Jet, Plasma };

	LFDisp::Method method = LFDisp::Method::DistgDisp;
	Color color = Color::Gray;
	int patchSize = 196;
	int views = 5;
};

struct LFParams {
	LFParamsPath path;
	LFParamsCalibrate calibrate;
	LFParamsISP isp;
	LFParamsDynamic dynamic;
	LFParamsSAI sai;
	LFParamsRefocus refocus;
	LFParamsSR sr;
	LFParamsDE de;
};

#endif