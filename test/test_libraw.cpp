#include <cstdio>
#include <iostream>
#include <libraw/libraw.h>
#include <opencv2/opencv.hpp>
#include <string>
bool fileExists(const std::string &filename) {
	std::ifstream f(filename);
	return f.good();
}
void process_image(char *file) {
	// Let us create an image processor
	LibRaw iProcessor;

	// Open the file and read the metadata
	iProcessor.open_file(file);

	// The metadata are accessible through data fields of the class
	printf("Image size: %d x %d\n", iProcessor.imgdata.sizes.width,
		   iProcessor.imgdata.sizes.height);

	// Let us unpack the image
	iProcessor.unpack();

	// Convert from imgdata.rawdata to imgdata.image:
	iProcessor.raw2image();

	// And let us print its dump; the data are accessible through data fields of
	// the class
	for (int i = 0;
		 i < iProcessor.imgdata.sizes.iwidth * iProcessor.imgdata.sizes.iheight;
		 i++) {
		printf("i=%d R=%d G=%d B=%d G2=%d\n", i, iProcessor.imgdata.image[i][0],
			   iProcessor.imgdata.image[i][1], iProcessor.imgdata.image[i][2],
			   iProcessor.imgdata.image[i][3]);
	}

	// Finally, let us free the image processor for work with the next image
	iProcessor.recycle();
}
int main() {
	char path[] = "/Users/jax/code/LightFieldLab/build/input/MOD_0015.RAW";
	// process_image(path);

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<uint8_t> buffer = std::vector<uint8_t>(size);
	file.read(reinterpret_cast<char *>(buffer.data()), size);
	file.close();

	// 2. 交给 LibRaw 从 buffer 中解码（绕过文件格式判断）
	LibRaw raw;
	int ret = raw.open_buffer((void *)buffer.data(), buffer.size());
	printf("buffer size: %zu\n", buffer.size());
	if (ret != LIBRAW_SUCCESS) {
		std::cerr << "❌ LibRaw failed to open buffer: " << libraw_strerror(ret)
				  << std::endl;
		// return false;
	}

	// 3. 解包
	if (raw.unpack() != LIBRAW_SUCCESS) {
		std::cerr << "❌ Failed to unpack RAW buffer." << std::endl;
		// return false;
	}

	// 4. 设置参数
	raw.imgdata.params.use_camera_wb = 1;
	raw.imgdata.params.no_auto_bright = 1;
	raw.imgdata.params.output_bps = 8;
	raw.imgdata.params.gamm[0] = 1.0f;
	raw.imgdata.params.gamm[1] = 1.0f;

	// 5. 解码
	if (raw.dcraw_process() != LIBRAW_SUCCESS) {
		std::cerr << "❌ Failed to process RAW buffer." << std::endl;
		// return false;
	}

	// 6. 输出图像
	libraw_processed_image_t *img = raw.dcraw_make_mem_image();
	if (!img) {
		std::cerr << "❌ Failed to get processed image." << std::endl;
		// return false;
	}
	cv::Mat imageOut =
		cv::Mat(img->height, img->width, CV_8UC3, img->data).clone();
	LibRaw::dcraw_clear_mem(img);
	raw.recycle();

	return 0;
}
