import os
import torch

def load_checkpoint(model, checkpoint_path):
    """
    健壮的权重加载函数：
    1. 自动识别 .pth 或 .tar
    2. 自动处理 'state_dict' 键
    3. 自动去除 DataParallel 产生的 'module.' 前缀
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    print(f"正在加载权重: {checkpoint_path} ...")
    
    # 加载文件到 CPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 1. 提取 state_dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # 也许整个字典就是权重
            state_dict = checkpoint
    else:
        # 也许 checkpoint 本身就是 state_dict 对象
        state_dict = checkpoint

    # 2. 处理 'module.' 前缀 (多卡训练遗留问题)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # 去掉 'module.'
        else:
            name = k
        new_state_dict[name] = v

    # 3. 加载到模型
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("权重加载成功 (Strict Mode)！")
    except Exception as e:
        print(f"Strict 加载失败，尝试非 Strict 模式... 错误: {e}")
        model.load_state_dict(new_state_dict, strict=False)
        print("权重加载成功 (Non-Strict Mode)，请检查是否有关键层缺失！")

    return model

def export_to_onnx(model, onnx_path, angRes=9):
    model.eval() # 必须切换到评估模式
    
    # 创建一个 Dummy Input
    # 尺寸必须是 angRes 的倍数，例如 5x5 的光场，单视角 64x64 -> 输入 320x320
    # 这里假设单视角 64x64
    h_sai, w_sai = 64, 64
    h_in, w_in = h_sai * angRes, w_sai * angRes
    dummy_input = torch.randn(1, 1, h_in, w_in).cpu()
    
    print(f"开始导出 ONNX 到: {onnx_path}")
    print(f"Dummy Input Shape: {dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=17, # 推荐 11 或 13
        input_names=['input'],
        output_names=['disp'], # 输出是视差图
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'disp':  {0: 'batch', 2: 'height', 3: 'width'}
        }
    )
    print("ONNX 导出完成！")