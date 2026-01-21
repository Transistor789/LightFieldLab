#ifndef LFPARAMS_H
#define LFPARAMS_H

#include "colormatcher.h"
#include "json.hpp"
#include "lfcalibrate.h"
#include "lfde.h"
#include "lfisp.h"
#include "lfsr.h"
#include "utils.h"

#include <atomic>
#include <string>
#include <vector>

struct LFParamsPath {
	std::string lfp, sai, white, extractLUT, dehexLUT, srModel, deModel;
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

struct LFParamsDynamic {
	std::vector<int> cameraID;
	std::atomic<bool> exit = false;
	std::atomic<bool> isCapturing = false;
	std::atomic<bool> isProcessing = false;
	std::atomic<int> capFrameCount{0};
	std::atomic<int> procFrameCount{0};
	bool showLFP = true;
	bool showSAI = true;
	BayerPattern bayer;
	int bitDepth;
	int width, height;
};

struct LFParamsColorEq {};

struct LFParamsRefocus {
	int crop = 0;
	float alpha = 1.0f;
	float shift = 0.0f;
};

struct LFParamsSR {
	int scale = 2;
	int patchSize = 128;
	int views = 5;
	SRMethod method = SRMethod::NEAREST;
};

struct LFParamsDE {
	DEMethod method = DEMethod::DistgDisp;
	DEColor color = DEColor::Gray;
	int patchSize = 128;
	int views = 9;
};

struct LFParams {
	ImageFileType imageType = ImageFileType::Lytro;
	LFParamsImage image;
	LFParamsPath path;
	CalibrateConfig calibrate;
	IspConfig isp;
	LFParamsDynamic dynamic;
	LFParamsSAI sai;
	LFParamsRefocus refocus;
	LFParamsSR sr;
	LFParamsDE de;
};

NLOHMANN_JSON_SERIALIZE_ENUM(ImageFileType, {{ImageFileType::Lytro, "Lytro"},
											 {ImageFileType::Raw, "Raw"},
											 {ImageFileType::Normal, "Normal"}})

NLOHMANN_JSON_SERIALIZE_ENUM(BayerPattern, {{BayerPattern::NONE, "NONE"},
											{BayerPattern::RGGB, "RGGB"},
											{BayerPattern::GRBG, "GRBG"},
											{BayerPattern::GBRG, "GBRG"},
											{BayerPattern::BGGR, "BGGR"}})

NLOHMANN_JSON_SERIALIZE_ENUM(ExtractMethod, {{ExtractMethod::Contour, "Contour"},
											 {ExtractMethod::GrayGravity, "GrayGravity"},
											 {ExtractMethod::CCA, "CCA"},
											 {ExtractMethod::LOG_NMS, "LOG_NMS"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Orientation, {{Orientation::HORZ, "HORZ"}, {Orientation::VERT, "VERT"}})

NLOHMANN_JSON_SERIALIZE_ENUM(DpcMethod, {{DpcMethod::Diretional, "Diretional"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Device,
							 {{Device::CPU, "CPU"}, {Device::CPU_OPENMP_SIMD, "CPU_OPENMP_SIMD"}, {Device::GPU, "GPU"}})

NLOHMANN_JSON_SERIALIZE_ENUM(ColorEqualizeMethod, {{ColorEqualizeMethod::Reinhard, "Reinhard"},
												   {ColorEqualizeMethod::HistMatch, "HistMatch"},
												   {ColorEqualizeMethod::MKL, "MKL"},
												   {ColorEqualizeMethod::MVGD, "MVGD"},
												   {ColorEqualizeMethod::HM_MKL_HM, "HM_MKL_HM"},
												   {ColorEqualizeMethod::HM_MVGD_HM, "HM_MVGD_HM"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LFParamsPath, lfp, sai, white, extractLUT, dehexLUT, srModel, deModel)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CalibrateConfig, hexgridfit, genLUT, autoEstimate, diameter, bitDepth, views, bayer,
								   ceMethod, orientation)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IspConfig, dpcThreshold, rawnr_sigma_r, rawnr_sigma_s, lscExp, uvnr_sigma_r,
								   uvnr_sigma_s, colorEqMethod, ceClipLimit, ceGridSize, seFactor, enableRAW, enableBLC,
								   enableDPC, enableRawNR, enableLSC, enableAWB, enableDemosaic, enableCCM, enableGamma,
								   enableCSC, enableUVNR, enableColorEq, enableCE, enableSE, enableExtract, enableDehex,
								   benchmark, dpcMethod, device)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LFParams, imageType, path, calibrate, isp)

#endif