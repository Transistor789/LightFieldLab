# LightFieldLab (光场实验室)

![C++](https://img.shields.io/badge/C++-20-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=flat-square&logo=cmake&logoColor=white)
![License](https://img.shields.io/badge/License-GPL_v3-blue.svg?style=flat-square)

**LightFieldLab** 是一个基于 C++20 的光场数据处理与分析桌面应用（Qt6），覆盖原始光场数据解码、ISP 处理、标定、超分辨率与深度估计等常见任务。集成了 NVIDIA TensorRT 用于深度学习模型推理。

## 核心功能 (Features)

### 1. 基础光场处理 (Core Processing)
* **原始数据解码 (Raw Decoding)**: 支持 Lytro Illum 相机 LFR/RAW 格式光场数据的读取与解析 (`raw_decode`, `lfio`)。
* **图像信号处理 (ISP)**: 黑电平校正、缺陷像素校正、raw域降噪、镜头阴影校正、白平衡、去马赛克、色彩校正、Gamma、色度降噪、对比度均衡、锐化等完整管线，提供 CPU 基线、CPU 加速 (OpenMP+SIMD)、GPU (CUDA) 三套后端 (`lfisp`)。
* **相机标定 (Calibration)**: 提供微透镜阵列 (MLA) 中心提取、排序与六边形网格拟合算法 (`lfcalibrate`, `centers_extract`, `hexgrid_fit`)。
* **重聚焦 (Refocusing)**: 实现基于频域或空域的数字重聚焦 (`lfrefocus`)。
	* **全焦图像生成 (All-in-Focus)**: 从重聚焦堆栈融合生成全焦图像 (`lfrefocus`)。
	* **色彩均衡 (Color Equalization)**: 提供 Reinhard、直方图匹配、MKL、MVGD 等 6 种色彩一致性矫正算法 (`colorequalize`)。
* **在线采集 (Live Capture)**: 通过 OpenCV 支持 USB 相机实时采集光场数据（跨平台），Windows 下额外支持 FT602 工业相机 (`lfcapture`)。

### 2. AI 增强与推理 (AI Powered by TensorRT)
基于 TensorRT 进行 C++ 工程化部署与 FP16 加速：
* **光场超分辨率 (Super-Resolution)**:
    * **DistgSSR** (TPAMI 2022)，Mosaic 布局输入。
    * **EPIT** (ICCV 2023)，Stack 布局输入。
* **光场深度/视差估计 (Disparity Estimation)**:
    * **DistgDisp** (TPAMI 2022)。
    * **OACC-Net** (CVPR 2022)。
* **轻量级超分备选**: OpenCV `dnn_superres` 模块，支持 ESPCN / FSRCNN 模型。
* **推理引擎封装**: `TRTWrapper` 封装 TensorRT 10.x API，支持 ONNX 模型加载、Engine 序列化与动态形状设置。

### 3. 可视化与交互 (Visualization)
* 基于 **Qt 6** 开发的图形用户界面 (`ui/`)。
* 支持光场子孔径图像阵列的实时预览与交互操作。

### 4. 工具与工具链 (Tools)
* **INRIA 数据集批处理**: 批量处理 Lytro Illum 标准测试集，输出处理结果与耗时日志。
* **标定评估工具**: 对比标定结果与真值，计算 pitch MAE 与缺失率 (`tools/calibration_evaluator.cpp`)。
* **性能分析 (Profiler)**: 内置分阶段计时器，支持 CPU / Fast / GPU 三套管线的逐模块耗时统计。

## 环境依赖

支持 **Windows (MSVC)** 与 **Linux (GCC)**，以 Windows 为主要开发环境。需要编译器支持 **C++20**，**CMake >= 3.20**。

核心依赖库如下：

| 组件 | 版本要求 | 说明 |
| :--- | :--- | :--- |
| **Qt** | ![Qt](https://img.shields.io/badge/Qt-6.10.0-41CD52?style=flat-square&logo=qt&logoColor=white) | GUI 框架 (推荐 MSVC 2022) |
| **OpenCV** | ![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | 计算机视觉 (推荐开启 CUDA 支持) |
| **TensorRT** | ![TensorRT](https://img.shields.io/badge/TensorRT-10.12-76B900?style=flat-square&logo=nvidia&logoColor=white) | 深度学习推理加速 |
| **CUDA** | ![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white) | 并行计算框架 (含 cuDNN) |
| **OpenSSL** | ![OpenSSL](https://img.shields.io/badge/OpenSSL-3.5.1-721412?style=flat-square&logo=openssl&logoColor=white) | 加密支持 |
| **Eigen** | ![Eigen](https://img.shields.io/badge/Eigen-5.0.0-1528AD?style=flat-square&logoColor=white) | 线性代数与网格拟合 (New Semantic Versioning) |
| **OpenMP** | ![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-blueviolet?style=flat-square) | CPU 并行加速 |
| **ONNX Runtime** | 1.22 | 测试用 GPU 版 ONNX 推理 |

## 项目结构

```text
LightFieldLab/
├── core/               # 核心算法库
│   ├── lfmbase.h       # 光场 AI 模型基类
│   ├── trtwrapper.cpp  # TensorRT 推理接口封装
│   ├── distgssr.cpp    # DistgSSR 超分算法实现
│   ├── epit.cpp        # EPIT 超分算法实现
│   ├── distgdisp.cpp   # DistgDisp 深度估计算法实现
│   ├── lfcalibrate.cpp # 标定算法实现
│   ├── lfrefocus.cpp   # 重聚焦算法实现
│   └── ...
├── ui/                 # Qt 用户界面源码
├── data/               # 资源文件
│   ├── calibration/    # 标定中间数据 (bin/png)
│   └── *.json/lfr      # 示例光场数据
├── models/             # 模型导出工具
│   ├── *.onnx          # ONNX 模型文件
│   ├── *_Windows.engine        # 转换后的 TensorRT 推理引擎
│   └── *.pth/*.py      # 原始 PyTorch 权重与定义
├── tests/              # 单元测试 (GTest/独立可执行程序)
├── tools/              # 工具程序 (数据批处理、标定评估)
└── CMakeLists.txt      # CMake 构建脚本

```

## 构建与使用

### 1. 编译项目

```bash
mkdir build
cd build
cmake .. 
cmake --build . --config Release

```

### 2. 模型准备

本项目不包含 PyTorch 训练代码，仅包含推理部署代码。你需要将 PyTorch 模型转换为 TensorRT Engine：

1. 使用 `python/export_onnx_all.py` 将 `.pth` 权重导出为 `.onnx` 模型。
2. 使用 TensorRT 自带工具 `trtexec` 将 `.onnx` 转换为 `_Windows.engine` (推荐 FP16 模式)：
```bash
trtexec.exe --onnx=data/DistgSSR_2x_5x5.onnx --saveEngine=data/DistgSSR_2x_1x1x640x640_FP16_Windows.engine --fp16

```


3. 确保生成的 `_Windows.engine` 文件位于 `data/` 目录下，程序运行时会自动加载。

## 参考项目与致谢

本项目中的核心算法与架构设计参考或移植自以下优秀的开源项目与学术论文，特此致谢：

* **DistgSSR** & **DistgDisp**:
    * **Repositories**: [https://github.com/YingqianWang/DistgSSR](https://github.com/YingqianWang/DistgSSR) | [https://github.com/YingqianWang/DistgDisp](https://github.com/YingqianWang/DistgDisp)
    * **Reference**: Wang, Y., Wang, L., Wu, G., Yang, J., An, W., Yu, J., & Guo, Y. (2022). **"Disentangling Light Fields for Super-Resolution and Disparity Estimation"**. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.

* **OACC-Net**:
    * **Repository**: https://github.com/YingqianWang/OACC-Net
    * **Reference**: Wang, Y., Wang, L., Liang, Z., Yang, J., An, W., & Guo, Y. (2022). **"Occlusion-Aware Cost Constructor for Light Field Depth Estimation"**.Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19809–19818.

* **EPIT (Updated to ICCV 2023 Work)**:
    * **Repository**: [https://github.com/ZhengyuLiang24/EPIT](https://github.com/ZhengyuLiang24/EPIT)
    * **Reference**: Liang, Z., Wang, Y., Wang, L., Yang, J., Zhou, S., & Guo, Y. (2023). **"Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution"**. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 12376-12386.

* **PlenoptiCam**:
    * **Repository**: [https://github.com/hahnec/plenopticam](https://github.com/hahnec/plenopticam)
    * **Reference**: Hahne, C., & Aggoun, A. (2021). **"PlenoptiCam v1.0: A Light-Field Imaging Framework"**. *IEEE Transactions on Image Processing (TIP)*, vol. 30, pp. 6757-6771.

## 许可证 (License)

本项目遵循 **GNU GPL v3** 许可证，完整条款见 [LICENSE](LICENSE)。

Copyright (C) 2025 LightFieldLab Contributors
