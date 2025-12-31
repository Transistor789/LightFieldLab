#ifndef LFPARAMS_H
#define LFPARAMS_H

#include <string>
#include <vector>

enum class BayerPattern { NONE, RGGB, GRBG, GBRG, BGGR };

struct LFParamsSource {
	std::string pathLFP, pathSAI, pathWhite, pathExtract, pathDehex;
	int width, height, bitDepth;
	BayerPattern bayer = BayerPattern::GRBG;

	int white_level = 1023, black_level = 64;
	std::vector<float> awb_gains = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> ccm_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
									 0.0f, 0.0f, 0.0f, 1.0f};
	float gamma = 2.2f;
};

struct LFParamsSAI {
	int width;
	int height;
	int bit;
	int row;
	int col;
};

struct LFParamsCalibrate {
	bool useCCA = false;
	bool saveLUT = false;
	bool gridfit = false;
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

	enum class DPCType { Dirctional, COUNT };
	DPCType dpcType = DPCType::Dirctional;
	int dpcThreshold = 100;
	float lscExp = 1.0;
	enum class DemosaicType { Bilinear, Gray, VGN, EA };
	DemosaicType demosaicType = DemosaicType::Bilinear;
};

struct LFParamsRefocus {
	int crop = 1;
	float alpha = 2.0;
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
	LFParamsSource source;
	LFParamsCalibrate calibrate;
	LFParamsISP isp;
	LFParamsRefocus refocus;
	LFParamsSR sr;
	LFParamsDE de;
};

#endif