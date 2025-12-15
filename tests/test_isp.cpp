#include "config.h"
#include "lfcalibrate.h"
#include "lfio.h"
#include "lfisp.h"
#include "utils.h"

#include <cstdint>
#include <format>
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

void test_isp() {
	LFIO load;
	auto lf_img = load.read_image("../data/toy.lfr");
	auto wht_img = load.read_image("../data/MOD_0015.RAW");
	// lf_img.convertTo(lf_img, CV_8U, 255.0 / 1023.0);
	// wht_img.convertTo(wht_img, CV_8U, 255.0 / 1023.0);

	ISPConfig config;
	config.bit = Config::Get().img_meta()["bit"].get<int>();
	config.black_level = Config::Get()
							 .img_meta()["blc"]["black"]
							 .get<std::vector<uint16_t>>()[0];
	config.white_level = Config::Get()
							 .img_meta()["blc"]["white"]
							 .get<std::vector<uint16_t>>()[0];
	// config.bit = 8;
	// config.black_level = 16;
	// config.white_level = 255;
	config.awb_gains =
		Config::Get().img_meta()["awb"].get<std::vector<float>>();
	config.ccm_matrix =
		Config::Get().img_meta()["ccm"].get<std::vector<float>>();

	LFIsp<uint16_t> isp(config, lf_img, wht_img);
	// LFIsp<uint8_t> isp(config, lf_img, wht_img);

	Timer timer;
	// isp.blc();
	// isp.blc_fast();

	// isp.dpc();
	// isp.dpc_fast();

	// isp.lsc();
	// isp.lsc_fast();
	// isp.lsc().demosaic();
	// isp.lsc_fast().demosaic();

	// isp.awb();
	// isp.awb_fast();
	// isp.awb().demosaic();
	// isp.awb_fast().demosaic();

	// isp.raw_process();
	// isp.raw_process_fast();

	// isp.blc().dpc().lsc().awb().demosaic();
	// isp.blc_fast().dpc_fast().lsc_fast().awb_fast().demosaic();

	// isp.raw_process().demosaic();
	// isp.raw_process_fast().demosaic();
	// isp.blc_fast().dpc_fast().lsc_awb_fused_fast().demosaic();

	// isp.lsc_awb_fused_fast().demosaic();

	isp.preview(1.5);
	// isp.updateImage(lf_img).preview().getPreviewResult();

	// isp.blc_fast().dpc_fast().lsc_fast().awb_fast().demosaic().ccm();
	// isp.blc_fast().dpc_fast().lsc_fast().awb_fast().demosaic().ccm_fast();

	// lf_img.convertTo(lf_img, CV_8UC(lf_img.channels()), 255.0 / 1023.0);
	// cv::demosaicing(lf_img, lf_img, cv::COLOR_BayerGR2RGB);

	timer.stop();
	timer.print_elapsed_ms();
	// imshowRaw("", isp.getResult());
	// cv::waitKey();

	// saveAs8Bit("../../data/fast_review.png", isp.getResult());
	// saveAs8Bit("../../data/fast_review.png", lf_img);

	cv::imwrite("../../data/preview.png", isp.getPreviewResult());

	// cv::imwrite("../../data/fast_review.png", lf_img);
	// cv::imshow("", isp.getResult());
	// cv::waitKey();

	// GpuIspPipeline<uint16_t> isp_gpu(config, lf_img, wht_img);
	// timer.start();
	// isp_gpu.blc().dpc().lsc().awb().demosaic();
	// timer.stop();
	// timer.print_elapsed_ms();

	// bool save = true;
	// if (save) {
	// 	saveAs8Bit("../../data/toy_demosaic_cpu.png", isp.getResult());
	// 	// cv::Mat result;
	// 	// isp_gpu.getResult(result);
	// 	// saveAs8Bit("../../data/toy_demosaic_gpu.png", result);
	// }
	// Timer timer;
	// isp.blc();
	// timer.stop();
	// timer.print_elapsed_ms("blc");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_blc.png", isp.getResult());
	// }

	// timer.start();
	// isp.dpc();
	// timer.stop();
	// timer.print_elapsed_ms("dpc");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_dpc.png", isp.getResult());
	// }

	// timer.start();
	// isp.lsc();
	// timer.stop();
	// timer.print_elapsed_ms("lsc");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_lsc.png", isp.getResult());
	// }

	// timer.start();
	// isp.awb();
	// timer.stop();
	// timer.print_elapsed_ms("awb");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_awb.png", isp.getResult());
	// }

	// timer.start();
	// isp.demosaic();
	// timer.stop();
	// timer.print_elapsed_ms("demosaic");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_demosaic.png", isp.getResult());
	// }

	// timer.start();
	// isp.ccm();
	// timer.stop();
	// timer.print_elapsed_ms("ccm");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_ccm.png", isp.getResult());
	// }

	// timer.start();
	// isp.gc();
	// timer.stop();
	// timer.print_elapsed_ms("gc");
	// if (save) {
	// 	saveAs8Bit("../../data/toy_gc.png", isp.getResult());
	// }
}
int main() {
	test_isp();

	return 0;
}