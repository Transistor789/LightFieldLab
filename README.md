# LightFieldLab (光场实验室)

![C++](https://img.shields.io/badge/C++-20-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=flat-square&logo=cmake&logoColor=white)
![License](https://img.shields.io/badge/License-GPL_v3-blue.svg?style=flat-square)

**LightFieldLab** 是一个基于现代 C++ (C++20) 开发的高性能光场数据处理与分析平台。本项目旨在提供从原始光场数据解码到高级计算机视觉应用的全流程解决方案，深度集成了 **NVIDIA TensorRT**，实现了光场超分辨率与深度估计算法的实时推理。

## ✨ 核心功能 (Features)

### 1. 基础光场处理 (Core Processing)
* **原始数据解码 (Raw Decoding)**: 支持 LFR/RAW 格式光场数据的读取与解析 (`raw_decode`, `lfio`)。
* **图像信号处理 (ISP)**: 包含去马赛克 (Demosaic)、白平衡、色彩校正与 Gamma 变换 (`lfisp`)。
* **相机标定 (Calibration)**: 提供微透镜阵列 (MLA) 中心提取、排序与六边形网格拟合算法 (`lfcalibrate`, `centers_extract`, `hexgrid_fit`)。
* **重聚焦与重采样 (Refocusing & Resampling)**: 实现基于频域或空域的数字重聚焦与光场重采样 (`lfrefocus`, `lfresample`)。

### 2. AI 增强与推理 (AI Powered by TensorRT)
本项目利用 TensorRT 对多个顶尖光场深度学习模型进行了 C++ 工程化部署与 FP16 加速：
* **光场超分辨率 (Super-Resolution)**:
    * 集成 **DistgSSR** (CVPR 2021) 模型。
    * 集成 **EPIT** (NeurIPS 2024) 模型。
    * 支持 Mosaic 与 Stack 两种数据输入布局。
* **光场深度/视差估计 (Disparity Estimation)**:
    * 集成 **DistgDisp** (TPAMI 2022) 模型，支持高精度的视差图计算。
* **高性能推理引擎**: 封装了 `TRTWrapper`，支持 ONNX 模型加载与 Engine 序列化/反序列化。

### 3. 可视化与交互 (Visualization)
* 基于 **Qt 6** 开发的图形用户界面 (`ui/`)。
* 支持光场子孔径图像阵列的实时预览与交互操作。

## 🛠️ 环境依赖 (Dependencies)

本项目主要在 **Windows (MSVC)** 环境下开发与测试，要求编译器支持 **C++20** 标准，且 **CMake >= 3.20**。

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
## 📂 项目结构

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
│   ├── opencv_srmodel/ # OpenCV自带超分模型
│   ├── *.engine        # 转换后的 TensorRT 推理引擎
│   ├── *.onnx          # ONNX 模型文件
│   └── *.json/lfr      # 示例光场数据
├── python/             # 模型导出工具
│   ├── export_onnx_all.py # PyTorch 模型转 ONNX 脚本
│   └── *.pth/*.py      # 原始 PyTorch 权重与定义
├── tests/              # 单元测试 (GTest/独立可执行程序)
└── CMakeLists.txt      # CMake 构建脚本

```

## 🚀 构建与使用

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
2. 使用 TensorRT 自带工具 `trtexec` 将 `.onnx` 转换为 `.engine` (推荐 FP16 模式)：
```bash
trtexec.exe --onnx=data/DistgSSR_2x_5x5.onnx --saveEngine=data/DistgSSR_2x_1x1x640x640_FP16.engine --fp16

```


3. 确保生成的 `.engine` 文件位于 `data/` 目录下，程序运行时会自动加载。

## 🔗 参考项目与致谢

本项目中的核心算法与架构设计参考或移植自以下优秀的开源项目与学术论文，特此致谢：

* **DistgSSR** & **DistgDisp**:
    * **Repositories**: [https://github.com/YingqianWang/DistgSSR](https://github.com/YingqianWang/DistgSSR) | [https://github.com/YingqianWang/DistgDisp](https://github.com/YingqianWang/DistgDisp)
    * **Reference**: Wang, Y., Wang, L., Wu, G., Yang, J., An, W., Yu, J., & Guo, Y. (2022). **"Disentangling Light Fields for Super-Resolution and Disparity Estimation"**. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.

* **EPIT (Updated to ICCV 2023 Work)**:
    * **Repository**: [https://github.com/ZhengyuLiang24/EPIT](https://github.com/ZhengyuLiang24/EPIT)
    * **Reference**: Liang, Z., Wang, Y., Wang, L., Yang, J., Zhou, S., & Guo, Y. (2023). **"Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution"**. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 12376-12386.

* **PlenoptiCam**:
    * **Repository**: [https://github.com/hahnec/plenopticam](https://github.com/hahnec/plenopticam)
    * **Reference**: Hahne, C., & Aggoun, A. (2021). **"PlenoptiCam v1.0: A Light-Field Imaging Framework"**. *IEEE Transactions on Image Processing (TIP)*, vol. 30, pp. 6757-6771.

## 📄 许可证 (License)

本项目遵循 **GNU GPL v3** 许可证，与 PlenoptiCam 保持一致。

```text
Copyright (C) 2025 LightFieldLab Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

```
