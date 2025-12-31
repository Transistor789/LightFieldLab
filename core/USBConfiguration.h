// USBConfiguration.h

#ifndef USBCONFIGURATION_H // 建议加上标准的include guard
#define USBCONFIGURATION_H

// ================= 修改开始 =================
#if defined(_WIN32) || defined(_WIN64)
// Windows 环境：保留原有的 DLL 导出/导入逻辑
#ifdef USB_CONFIGURATION_EXPORTS
#define USB_CONFIGURATION_API __declspec(dllexport)
#else
#define USB_CONFIGURATION_API __declspec(dllimport)
#endif
#else
// Linux 环境：定义为空，变成普通函数声明
#define USB_CONFIGURATION_API
#endif
// ================= 修改结束 =================

#ifdef __cplusplus
extern "C" {
#endif

USB_CONFIGURATION_API int SetUSBConfiguration(int ImageWidth, int ImageHeight,
											  unsigned char ColorSpace,
											  unsigned char BayerFormat,
											  unsigned char CameraTap,
											  unsigned char BitWidth,
											  bool UVChange, bool HSDE);

USB_CONFIGURATION_API int GetRealImageData(
	unsigned char* OriginalImageData, unsigned char* DstImageData,
	int OriginalWidth, int OriginalHeight, int DstWidth, int DstHeight,
	unsigned char ColorSpace, unsigned char BitWidth);

#ifdef __cplusplus
}
#endif

#endif // USBCONFIGURATION_H