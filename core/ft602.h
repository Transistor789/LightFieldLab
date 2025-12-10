#ifndef FT602_H_EKYVTQBG
#define FT602_H_EKYVTQBG

#define CHANNEL_COUNT		4
#define RESOLUTION_COUNT	14

#include <cstdint>

typedef void* FT_HANDLE;

enum _FT_STATUS {
	FT_OK,
	FT_INVALID_HANDLE,
	FT_DEVICE_NOT_FOUND,
	FT_DEVICE_NOT_OPENED,
	FT_IO_ERROR,
	FT_INSUFFICIENT_RESOURCES,
	FT_INVALID_PARAMETER, /* 6 */
	FT_INVALID_BAUD_RATE,
	FT_DEVICE_NOT_OPENED_FOR_ERASE,
	FT_DEVICE_NOT_OPENED_FOR_WRITE,
	FT_FAILED_TO_WRITE_DEVICE, /* 10 */
	FT_EEPROM_READ_FAILED,
	FT_EEPROM_WRITE_FAILED,
	FT_EEPROM_ERASE_FAILED,
	FT_EEPROM_NOT_PRESENT,
	FT_EEPROM_NOT_PROGRAMMED,
	FT_INVALID_ARGS,
	FT_NOT_SUPPORTED,

	FT_NO_MORE_ITEMS,
	FT_TIMEOUT, /* 19 */
	FT_OPERATION_ABORTED,
	FT_RESERVED_PIPE,
	FT_INVALID_CONTROL_REQUEST_DIRECTION,
	FT_INVALID_CONTROL_REQUEST_TYPE,
	FT_IO_PENDING,
	FT_IO_INCOMPLETE,
	FT_HANDLE_EOF,
	FT_BUSY,
	FT_NO_SYSTEM_RESOURCES,
	FT_DEVICE_LIST_NOT_READY,
	FT_DEVICE_NOT_CONNECTED,
	FT_INCORRECT_DEVICE_PATH,

	FT_OTHER_ERROR,
};

typedef unsigned long FT_STATUS;
#define FT_SUCCESS(status) ((status) == FT_OK)
#define FT_FAILED(status)  ((status) != FT_OK)

//
// Create flags
//
#define FT_OPEN_BY_SERIAL_NUMBER                            0x00000001
#define FT_OPEN_BY_DESCRIPTION                              0x00000002
#define FT_OPEN_BY_LOCATION                                 0x00000004
#define FT_OPEN_BY_GUID                                     0x00000008
#define FT_OPEN_BY_INDEX									0x00000010



enum FT_BURST_BUFFER {
	FT_BURST_OFF = 0,
	FT_BURST_2K = 1,
	FT_BURST_4K = 2,
	FT_BURST_8K = 4,
	FT_BURST_16K = 8,
};

enum FT_FIFO_BUFFER {
	FT_FIFO_CLOSED = 0,
	FT_FIFO_2K,
	FT_FIFO_4K,
	FT_FIFO_6K,
	FT_FIFO_8K,
	FT_FIFO_10K,
	FT_FIFO_12K,
	FT_FIFO_14K,
	FT_FIFO_16K,
};

enum FT_PIPE_DIRECTION {
	FT_PIPE_DIR_IN,
	FT_PIPE_DIR_OUT,
	FT_PIPE_DIR_COUNT,
};


struct FT_CHIP_FLAGS {
	uint8_t fifo_clk : 2;
	uint8_t fifo_600mode : 1;
	uint8_t usb_self_powered : 1;
	uint8_t enable_remote_wakeup : 1;
	uint8_t enable_fifo_clock_during_sleep : 1;
	uint8_t disable_chip_powerdown : 1;

	uint8_t reserved1 : 1;

	uint8_t gpio0_is_output_pin : 1;
	uint8_t gpio0_pull_control : 2;
	uint8_t gpio0_schmitt_trigger : 1;

	uint8_t gpio1_is_output_pin : 1;
	uint8_t gpio1_pull_control : 2;
	uint8_t gpio1_schmitt_trigger : 1;

	uint8_t gpio2_is_output_pin : 1;
	uint8_t gpio2_pull_control : 2;
	uint8_t gpio2_schmitt_trigger : 1;
	uint8_t gpio2_interrupt_enabled : 1;

	/* GPIO[0:1] output leve when enable_battery_charging is set
	 * Not connected: 00
	 * Standard downstream port (SDP): 01
	 * Charging downstream port (CDP): 10
	 * Dedicated charging port (DCP): 11
	*/
	uint8_t enable_battery_charging : 1;

	uint8_t reserved2 : 2;
	uint8_t reserved3;
};

struct FT_USB_STRING_DESCRIPTOR {
	unsigned char length;
	unsigned char descriptor_type; /* Must set to 0x3 */
	unsigned short string[31];
};

struct FT_60XCOMMON_CONFIGURATION {
	unsigned short length;
	/* Generic USB configs */
	//unsigned short max_power_cs;
	unsigned char max_power_cs;
	unsigned char max_power_ss;
	unsigned short vendor_id;
	unsigned short product_id;

	struct FT_CHIP_FLAGS chip;

	/* FIFO & EPC settings */
	/* enum FT_FIFO_BUFFER */
	unsigned char fifo[FT_PIPE_DIR_COUNT][4];
	/* enum FT_BURST_BUFFER */
	unsigned char epc[FT_PIPE_DIR_COUNT][4];
	char version[16];
};


enum I2C_SPEED {
	I2C_DISABLED,
	I2C_1MHZ,
	I2C_400KHZ,
	I2C_100KHZ,
};

enum FT602_STRING_DESC_TYPE {
	FT602_STR_DESC_LANGID_ARRAY,
	FT602_STR_DESC_MANUFACTURER,
	FT602_STR_DESC_PRODUCT,
	FT602_STR_DESC_CHANNEL1,
	FT602_STR_DESC_CHANNEL2,
	FT602_STR_DESC_CHANNEL3,
	FT602_STR_DESC_CHANNEL4,
	FT602_STR_DESC_SERIALNUMBER,
	FT602_STR_DESC_COUNT,
	FT602_STR_DESC_OTHER_COUNT = FT602_STR_DESC_CHANNEL4,
};

enum USB_SPEED_TYPE {
	USB_SPEED_FS,
	USB_SPEED_HS,
	USB_SPEED_SS,
	USB_SPEED_COUNT,
};

enum CONTROL_TYPE {
	/* Order at firmware side is default, resolution, minimal, maximum */
	TYPE_NONE = 0,
	TYPE_DEF = 1,
	TYPE_DEF_RES = 2,
	TYPE_DEF_RES_MIN_MAX = 4,
};

struct FRAME_RESOLUTION {
	unsigned short wWidth;
	unsigned short wHeight;
	unsigned long dwPixelClock;
	unsigned long dwFrameInterval; /* 10000000 / fps */
};

struct FRAME_INFO {
	struct FRAME_RESOLUTION res[RESOLUTION_COUNT];

	unsigned long dwFourcc;

	unsigned char byBitsPerPixel;
	/* Color matching, UVC 1.1 p68 */
	unsigned char bColorPrimaries;
	unsigned char bTransferCharacteristics;
	unsigned char bMatrixCoefficients;
};

struct UVC_CONTROL_CONFIG {
	unsigned long info_get_support; /* bitmap of enum UVC_REGS */
	unsigned long info_set_support; /* bitmap of enum UVC_REGS */

	/* Camera Terminal bmControls Bitmaps:
		D0: Scanning Mode - not supported
		D1: Auto-Exposure Mode
		D2: Auto-Exposure Priority
		D3: Exposure Time (Absolute)
		D4: Exposure Time (Relative)
		D5: Focus (Absolute)
		D6 : Focus (Relative)
		D7: Iris (Absolute)
		D8 : Iris (Relative)
		D9: Zoom (Absolute)
		D10: Zoom (Relative)
		D11: PanTilt (Absolute)
		D12: PanTilt (Relative)
		D13: Roll (Absolute)
		D14: Roll (Relative)
		D15: Reserved
		D16: Reserved
		D17: Focus, Auto
		D18: Privacy - not supported */
	unsigned char bmCameraTerminalControls[3];
	/* Process Unit bmControls Bitmaps:
		D0: Brightness
		D1: Contrast
		D2: Hue
		D3: Saturation
		D4: Sharpness
		D5: Gamma
		D6: White Balance Temperature
		D7: White Balance Component
		D8: Backlight Compensation
		D9: Gain
		D10: Power Line Frequency
		D11: Hue, Auto
		D12: White Balance Temperature, Auto
		D13: White Balance Component, Auto
		D14: Digital Multiplier
		D15: Digital Multiplier Limit
		D16: Analog Video Standard
		D17: Analog Video Lock Status - not supported */
	unsigned char bmProcessUnitControls[3];
	/* bmVideoStandards bitmaps:
		D0: None
		D1: NTSC – 525/60
		D2: PAL – 625/50
		D3: SECAM – 625/50
		D4: NTSC – 625/50
		D5: PAL – 525/60 */
	unsigned char bmVideoStandards;
	/* Start of default value of UVC controls */
	unsigned char bAutoFocusMode;

	unsigned long dwExposureTimeAbsolute[TYPE_DEF_RES_MIN_MAX];
	unsigned short wFocusAbsolute[TYPE_DEF_RES_MIN_MAX];
	struct {
		char bFocusRelatve;
		unsigned char bSpeed;
	} stFocusRelative[TYPE_DEF_RES_MIN_MAX];
	unsigned short wIrisAbsolute[TYPE_DEF_RES_MIN_MAX];
	unsigned short wObjectiveFocalLength[TYPE_DEF_RES_MIN_MAX];
	struct {
		char bZoom;
		bool bDigitalZoom;
		unsigned char bSpeed;
	} stZoomRelative[TYPE_DEF_RES_MIN_MAX];
	struct {
		int32_t dwPanAbsolute;
		int32_t dwTiltAbsolute;
	} stPanTiltAbsolute[TYPE_DEF_RES_MIN_MAX];
	struct {
		char bPanRelative;
		unsigned char bPanSpeed;
		char bTiltRelatve;
		unsigned char bTiltSpeed;
	} stPanTiltRelative[TYPE_DEF_RES_MIN_MAX];
	int16_t wRollAbsolute[TYPE_DEF_RES_MIN_MAX];
	struct {
		char bRollRelative;
		unsigned char bSpeed;
	} stRollRelative[TYPE_DEF_RES_MIN_MAX];
	unsigned short wBacklightCompensation[TYPE_DEF_RES_MIN_MAX];
	int16_t wBrightness[TYPE_DEF_RES_MIN_MAX];
	unsigned short wContrast[TYPE_DEF_RES_MIN_MAX];
	unsigned short wGain[TYPE_DEF_RES_MIN_MAX];
	int16_t wHue[TYPE_DEF_RES_MIN_MAX];
	unsigned short wSaturation[TYPE_DEF_RES_MIN_MAX];
	unsigned short wSharpness[TYPE_DEF_RES_MIN_MAX];
	unsigned short wGamma[TYPE_DEF_RES_MIN_MAX];
	unsigned short wWhiteBalanceTemperature[TYPE_DEF_RES_MIN_MAX];
	struct {
		unsigned short wWhiteBalanceBlue;
		unsigned short wWhiteBalanceRed;
	} stWhiteBalanceComponent[TYPE_DEF_RES_MIN_MAX];
	unsigned short wMultiplierStep[TYPE_DEF_RES_MIN_MAX];
	unsigned short wMultiplierLimit[TYPE_DEF_RES_MIN_MAX];
	unsigned char bAutoExposureMode[TYPE_DEF_RES];
	unsigned char bWhiteBalanceComponentAuto;
	unsigned char wWhiteBalanceTemperatureAuto;

	unsigned char bHueAuto;
	unsigned char bPowerLineFrequency;
	/* End of default value of UVC controls */
};

struct I2C_CONFIG {
	/* I2C must be enabled if:
	 * info_get_support != 0 || info_set_support != 0
	 * Defined more than 1 resolution in each FRAME_INFO */
	uint8_t speed : 2;

	uint8_t reserved : 1;
	/* 1 - 16, default 32ms (2^9-1)*125us */
	uint8_t interrupt_endpoint_interval : 5;

	uint8_t address;
};

struct _FT_602CONFIGURATION {
	struct FT_60XCOMMON_CONFIGURATION common;
	struct FT_USB_STRING_DESCRIPTOR desc[FT602_STR_DESC_COUNT];
	/* Second language only support manufacturer and product string */
	struct FT_USB_STRING_DESCRIPTOR desc_other[FT602_STR_DESC_OTHER_COUNT];

	struct FRAME_INFO frame[CHANNEL_COUNT][USB_SPEED_COUNT];
	struct UVC_CONTROL_CONFIG controls[CHANNEL_COUNT];
	struct I2C_CONFIG i2c; //2
};
typedef struct _FT_602CONFIGURATION FT_602CONFIGURATION, *PFT_602CONFIGURATION;


//
//
// FT602 Library related APIs and definititions
//
//

typedef enum _FT_FILTER_TYPE
{
	FT_FILTER_NONE = 0,
	FT_FILTER_VID = 1,
	FT_FILTER_PID = 2,
	FT_FILTER_DESCRIPTION = 4
}FT_FILTER_TYPE;

#ifdef FT602_EXPORTS
#define FT602_API __declspec(dllexport)
#elif defined(FT602_LIB)
#define FT602_API
#else
#define FT602_API __declspec(dllimport)
#endif

#pragma pack(1)
typedef struct _FT_SETUP_PACKET {
	unsigned char   RequestType;
	unsigned char   Request;
	unsigned short  Value;
	unsigned short  Index;
	unsigned short  Length;
} FT_SETUP_PACKET, * PFT_SETUP_PACKET;
#pragma pack()

//
// Notification callback type
//
typedef enum _E_FT_NOTIFICATION_CALLBACK_TYPE
{
	E_FT_NOTIFICATION_CALLBACK_TYPE_INTERRUPT = 2,
	E_FT_NOTIFICATION_CALLBACK_TYPE_INTERRUPT_STOPPED = 3,

} E_FT_NOTIFICATION_CALLBACK_TYPE;

//
// Notification callback function
//
typedef void(*FT_NOTIFICATION_CALLBACK)(void* pvCallbackContext,
	E_FT_NOTIFICATION_CALLBACK_TYPE eCallbackType, void* pvCallbackInfo);

#ifdef __cplusplus
extern "C" {
#endif

	FT602_API FT_STATUS __stdcall FT_Create(
		void* pvArg,
		unsigned long dwFlags,
		FT_HANDLE* pftHandle
	);

	FT602_API FT_STATUS __stdcall FT_Close(
		FT_HANDLE ftHandle
	);

	FT602_API FT_STATUS __stdcall FT_GetChipConfiguration(
		FT_HANDLE ftHandle,
		void* pvConfiguration
	);

	FT602_API FT_STATUS __stdcall FT_SetChipConfiguration(
		FT_HANDLE ftHandle,
		void* pvConfiguration
	);

	FT602_API FT_STATUS __stdcall FT_ControlTransfer(
		FT_HANDLE ftHandle,
		FT_SETUP_PACKET tSetupPacket,
		unsigned char* pucBuffer,
		unsigned long ulBufferLength,
		unsigned long* pulLengthTransferred
	);

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: FT602_H_EKYVTQBG */
