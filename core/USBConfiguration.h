#ifdef USB_CONFIGURATION_EXPORTS
#define USB_CONFIGURATION_API __declspec(dllexport)
#else
#define USB_CONFIGURATION_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif
USB_CONFIGURATION_API int SetUSBConfiguration(int ImageWidth, int ImageHeight, unsigned char ColorSpace, unsigned char BayerFormat, unsigned char CameraTap, unsigned char BitWidth, bool UVChange, bool HSDE);

USB_CONFIGURATION_API int GetRealImageData(unsigned char* OriginalImageData, unsigned char* DstImageData, int OriginalWidth, int OriginalHeight, int DstWidth, int DstHeight, unsigned char ColorSpace, unsigned char BitWidth);

#ifdef __cplusplus
}
#endif