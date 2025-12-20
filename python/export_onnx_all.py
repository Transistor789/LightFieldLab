import torch
import torch.nn as nn
import os
import argparse
from collections import OrderedDict

# 导入你的模型定义
# 确保 DistgSSR.py, EPIT.py, LFSSRNet.py 在同一目录下
from DistgSSR import Net as DistgSSRNet
from EPIT import get_model as EPITNet
from LFSSR import LFSSR

def load_weights(model, path, device):
    """通用的权重加载函数，处理 module. 前缀和不同的保存字典结构"""
    if not os.path.exists(path):
        print(f"[Error] Weight file not found: {path}")
        return False

    print(f"    Loading weights from: {path}")
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # 1. 提取 state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 2. 清洗 module. 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # 3. 加载
        model.load_state_dict(new_state_dict, strict=True)
        print("    [Success] Weights loaded.")
        return True
    except Exception as e:
        print(f"    [Failed] Load error: {e}")
        return False

def export_distgssr(scale, angRes, pth_path):
    print(f"\n=== Exporting DistgSSR ({scale}x, {angRes}x{angRes}) ===")
    
    # 1. 初始化
    device = torch.device("cpu")
    model = DistgSSRNet(angRes=angRes, factor=scale).to(device)
    model.eval()

    # 2. 加载权重
    if not load_weights(model, pth_path, device):
        return

    # 3. Dummy Input
    # DistgSSR 输入: [B, 1, angRes*H, angRes*W] (MacPI 大图模式)
    H, W = 32, 32
    # 注意：输入尺寸必须是 angRes 的倍数
    dummy_input = torch.randn(1, 1, H * angRes, W * angRes).to(device)
    
    save_name = f"DistgSSR_{scale}x_{angRes}x{angRes}.onnx"
    
    # 4. 导出
    torch.onnx.export(
        model, dummy_input, save_name,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'h_macpi', 3: 'w_macpi'},
            'output': {0: 'batch', 2: 'out_h', 3: 'out_w'}
        }
    )
    print(f"    -> Exported to {save_name}")

def export_epit(scale, angRes, pth_path):
    print(f"\n=== Exporting EPIT ({scale}x, {angRes}x{angRes}) ===")

    # 1. 初始化
    device = torch.device("cpu")
    model = EPITNet(angRes=angRes, scale=scale).to(device)
    model.eval()

    # 2. 加载权重
    if not load_weights(model, pth_path, device):
        return

    # 3. Dummy Input
    # EPIT 输入: [B, 1, U, V, H, W] (6D Tensor, Y Only)
    H, W = 32, 32
    dummy_input = torch.randn(1, 1, angRes, angRes, H, W).to(device)
    
    save_name = f"EPIT_{scale}x_{angRes}x{angRes}.onnx"

    # 4. 导出
    torch.onnx.export(
        model, dummy_input, save_name,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 4: 'h', 5: 'w'}, # 动态维度在最后两维
            'output': {0: 'batch', 4: 'out_h', 5: 'out_w'}
        }
    )
    print(f"    -> Exported to {save_name}")

def export_lfssr(scale, angRes, pth_path):
    print(f"\n=== Exporting LFSSRNet ({scale}x, {angRes}x{angRes}) ===")

    # 1. 初始化 Mock 参数
    class MockOpt:
        def __init__(self):
            self.angular_num = angRes
            self.scale = scale
            self.feature_num = 64
            self.layer_num_refine = 3 # 针对 7x7 权重推断的配置
            # 针对 7x7 权重的特殊层数配置，基于之前的报错分析
            if angRes == 7:
                 self.layer_num = [5, 2, 2, 3] 
            else:
                 self.layer_num = [2, 3, 3, 2] # 5x5 的默认配置

    opt = MockOpt()
    device = torch.device("cpu")
    model = LFSSR(opt).to(device)
    model.eval()

    # 2. 加载权重
    if not load_weights(model, pth_path, device):
        return

    # 3. Dummy Input
    # LFSSR 输入: [B, angRes^2, H, W] (Stack 模式, Y Only)
    # 这里的 Channel 维度实际上是 angRes*angRes 个视点堆叠
    H, W = 32, 32
    dummy_input = torch.randn(1, angRes * angRes, H, W).to(device)
    
    save_name = f"LFSSRNet_{scale}x_{angRes}x{angRes}.onnx"

    # 4. 导出
    torch.onnx.export(
        model, dummy_input, save_name,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'h', 3: 'w'},
            'output': {0: 'batch', 2: 'out_h', 3: 'out_w'}
        }
    )
    print(f"    -> Exported to {save_name}")

if __name__ == "__main__":
    # ==========================================
    # 1. 转换 DistgSSR (5x5)
    # ==========================================
    # 文件列表显示是 .pth.tar
    # export_distgssr(scale=2, angRes=5, pth_path="python/DistgSSR_2x_5x5.pth.tar")
    # export_distgssr(scale=4, angRes=5, pth_path="python/DistgSSR_4x_5x5.pth.tar")

    # ==========================================
    # 2. 转换 EPIT (5x5)
    # ==========================================
    # export_epit(scale=2, angRes=5, pth_path="python/EPIT_2x_5x5.pth")
    # export_epit(scale=4, angRes=5, pth_path="python/EPIT_4x_5x5.pth")

    # ==========================================
    # 3. 转换 LFSSRNet (7x7)
    # ==========================================
    # 注意：文件名显示 LFSSRNet 是 7x7 的权重
    # export_lfssr(scale=2, angRes=7, pth_path="python/LFSSRNet_2x_7x7.pth")
    export_lfssr(scale=4, angRes=7, pth_path="python/LFSSRNet_4x_7x7.pth")
    
    print("\nAll tasks finished.")