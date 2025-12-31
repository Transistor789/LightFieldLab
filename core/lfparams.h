#ifndef LFPARAMS_H
#define LFPARAMS_H

#include <atomic>
#include <string>
#include <vector>

enum class BayerPattern { NONE, RGGB, GRBG, GBRG, BGGR };

struct LFParamsPath {
	std::string lfp, sai, white, extractLUT, dehexLUT;
};

struct LFParamsSAI {
	int row = 5;
	int col = 5;
	int rows = 9;
	int cols = 9;
	bool isPlaying = false;
};

struct LFParamsCalibrate {
	bool useCCA = false;
	bool autoEstimate = false;
	bool saveLUT = false;
	bool gridFit = false;
	int views = 9;
};

struct LFParamsISP {
	enum class DPCType { Dirctional, COUNT };
	enum class DemosaicType { Bilinear, Gray, VGN, EA };

	int width, height, bitDepth;
	BayerPattern bayer = BayerPattern::GRBG;

	int white_level = 1023, black_level = 64;
	std::vector<float> awb_gains = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> ccm_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
									 0.0f, 0.0f, 0.0f, 1.0f};
	float gamma = 2.2f;

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

	DPCType dpcType = DPCType::Dirctional;
	int dpcThreshold = 100;
	float lscExp = 1.0;
	DemosaicType demosaicType = DemosaicType::Bilinear;
};

struct LFParamsDynamic {
	std::vector<int> cameraID;
	std::atomic_bool isCapturing = false;
	std::atomic_bool isProcessing = false;
};

struct LFParamsRefocus {
	int crop = 0;
	float alpha = 1.0;
};

struct LFParamsSR {
	enum class Type {
		NEAREST,
		LINEAR,
		CUBIC,
		LANCZOS,
		ESPCN,
		FSRCNN,
		DISTGSSR
	};
	int scale = 2;
	int patchSize = 196;
	int views = 5;
	Type type = Type::NEAREST;
};

struct LFParamsDE {
	enum class Type { DistgSSR };
	enum class Color { Gray, Jet, Plasma };

	Type type = Type::DistgSSR;
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