import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.angRes = angRes
        self.factor = factor
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False)
        self.disentg = CascadeDisentgGroup(n_group, n_block, angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = SAI2MacPI(x, self.angRes)
        buffer = self.init_conv(x)
        buffer = self.disentg(buffer)
        buffer_SAI = MacPI2SAI(buffer, self.angRes)
        out = self.upsample(buffer_SAI) + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels//4, channels//2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        return buffer + x


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        # 修改这里：使用 -1 自动计算宽度，防止导出时维度计算被写死
        y = x.view(b, c, h, -1)
        return y


# def MacPI2SAI(x, angRes):
#     out = []
#     for i in range(angRes):
#         out_h = []
#         for j in range(angRes):
#             out_h.append(x[:, :, i::angRes, j::angRes])
#         out.append(torch.cat(out_h, 3))
#     out = torch.cat(out, 2)
#     return out

# def SAI2MacPI(x, angRes):
#     b, c, hu, wv = x.shape
#     h, w = hu // angRes, wv // angRes
#     tempU = []
#     for i in range(h):
#         tempV = []
#         for j in range(w):
#             tempV.append(x[:, :, i::h, j::w])
#         tempU.append(torch.cat(tempV, dim=3))
#     out = torch.cat(tempU, dim=2)
#     return out

def SAI2MacPI(x, angRes):
    """
    SAI (Sub-Aperture Image) -> MacPI (Macropixel Image)
    把光场从“视点阵列”格式转换为“宏像素”格式
    Input: [B, C, angRes*H, angRes*W]
    Output: [B, C, H*angRes, W*angRes]
    """
    b, c, hu, wv = x.shape
    # 使用 -1 让 PyTorch 自动推导维度，增强动态兼容性
    h, w = hu // angRes, wv // angRes
    
    # 1. 拆分维度：把大图拆成 [angRes, h] 块
    # [B, C, angRes, h, angRes, w]
    x = x.view(b, c, angRes, h, angRes, w)
    
    # 2. 交换维度：把 h, w 移到前面，把 angRes 移到后面
    # 目标顺序: [B, C, h, angRes, w, angRes]
    x = x.permute(0, 1, 3, 2, 5, 4).contiguous()
    
    # 3. 合并维度：变成宏像素格式
    # [B, C, h*angRes, w*angRes]
    out = x.view(b, c, hu, wv)
    return out


def MacPI2SAI(x, angRes):
    """
    MacPI -> SAI
    把光场从“宏像素”格式转换为“视点阵列”格式
    Input: [B, C, H*angRes, W*angRes]
    Output: [B, C, angRes*H, angRes*W]
    """
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    
    # 1. 拆分维度：把宏像素拆开
    # [B, C, h, angRes, w, angRes]
    x = x.view(b, c, h, angRes, w, angRes)
    
    # 2. 交换维度：把 angRes 移到前面，形成视点阵列
    # 目标顺序: [B, C, angRes, h, angRes, w]
    x = x.permute(0, 1, 3, 2, 5, 4).contiguous()
    
    # 3. 合并维度
    # [B, C, angRes*h, angRes*w]
    out = x.view(b, c, hu, wv)
    return out

if __name__ == "__main__":
    import os
    
    # 1. 配置模型参数
    angRes = 5   # 角度分辨率 (5x5)
    factor = 2   # 超分倍率 (4x)
    output_onnx_name = f"DistgSSR_{factor}xSR_{angRes}x{angRes}.onnx"

    # 2. 初始化模型
    # 建议先在 CPU 上导出，兼容性更好；如果报错再尝试 .cuda()
    device = torch.device("cpu") 
    model = Net(angRes=angRes, factor=factor).to(device)
    model.eval()

    # 加载权重 (如果有训练好的 .pth 文件，请取消注释并修改路径)
    # model_path = "path/to/your/model.pth"
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict)

    # 3. 构造 Dummy Input (样板输入)
    # 尺寸必须是 angRes 的倍数。
    # 这里假设输入是 SAI 阵列拼接图，格式为 [B, 1, angRes*H, angRes*W]
    H_dummy, W_dummy = 32, 32
    dummy_input = torch.randn(1, 1, H_dummy * angRes, W_dummy * angRes).to(device)

    # 4. 执行 ONNX 导出
    print(f"Starting export: {output_onnx_name} ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_name,
        export_params=True,        # 导出训练好的参数权重
        opset_version=14,          # 推荐 14，对动态尺寸和 resize 支持较好
        do_constant_folding=True,  # 优化常量折叠
        input_names=['input'],     # 输入节点名称
        output_names=['output'],   # 输出节点名称
        dynamic_axes={
            'input': {
                0: 'batch_size',   # Batch 维度动态
                2: 'height',       # 高度动态
                3: 'width'         # 宽度动态
            },
            'output': {
                0: 'batch_size',
                2: 'out_height',
                3: 'out_width'
            }
        }
    )
    print(f"Success! Model saved to {output_onnx_name}")
    print("\n[Check Dimensions]")
    print(f"Dummy Input Shape: {dummy_input.shape}")
    # 简单的验证性推理
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Dummy Output Shape: {output.shape}")