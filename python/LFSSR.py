import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

# =========================================================================
# 1. 基础模块定义
# =========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out

def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)

class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()
        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, c, h, w = x.shape  
        # 注意：这里的 N 实际上是 Batch * an^2，因为前面做了 view(-1)
        # 为了支持动态尺寸，我们反向推导真正的 Batch Size
        # 但 AltFilter 的逻辑其实只依赖于把 channel 拆分重排
        
        # 这里的 reshape 逻辑：[Batch*An^2, C, H, W] -> [Batch, An^2, C, H*W]
        # 由于我们无法在 forward 里获得原始 Batch，只能依赖于输入的 N 必须是 an^2 的倍数
        real_N = N // (self.an * self.an)

        out = self.relu(self.spaconv(x))
        
        # Spatial -> Angular 变换
        out = out.view(real_N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.view(real_N * h * w, c, self.an, self.an)

        out = self.relu(self.angconv(out))

        # Angular -> Spatial 变换
        out = out.view(real_N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(real_N * self.an * self.an, c, h, w)

        return out

# =========================================================================
# 2. 主网络结构 (LFSSRNet) - Y Only
# =========================================================================

class LFSSR(nn.Module):
    def __init__(self, opt):
        super(LFSSR, self).__init__()
        
        # 参数配置
        fn = opt.feature_num
        self.an = opt.angular_num
        self.an2 = self.an * self.an
        self.scale = opt.scale

        # === 定义网络层 (保持不变) ===
        self.fea_conv0 = nn.Conv2d(1, fn, 3, 1, 1, bias=True)
        self.fea_resblock = make_layer(ResidualBlock, fn, opt.layer_num[0])

        self.pair_conv0 = nn.Conv2d(2 * fn, fn, 3, 1, 1, bias=True)
        self.pair_resblock = make_layer(ResidualBlock, fn, opt.layer_num[1])
        self.pair_conv1 = nn.Conv2d(fn, fn, 3, 1, 1, bias=True)

        self.fusion_view_conv0 = nn.Conv2d(self.an2, fn, 3, 1, 1, bias=True)
        self.fusion_view_resblock = make_layer(ResidualBlock, fn, opt.layer_num[2])
        self.fusion_view_conv1 = nn.Conv2d(fn, 1, 3, 1, 1, bias=True)

        self.fusion_fea_conv0 = nn.Conv2d(fn, fn, 3, 1, 1, bias=True)
        self.fusion_fea_resblock = make_layer(ResidualBlock, fn, opt.layer_num[3])

        up = []
        for _ in range(int(math.log(self.scale, 2))):
            up.append(nn.Conv2d(fn, 4 * fn, 3, 1, 1, bias=True))
            up.append(nn.PixelShuffle(2))
            up.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*up)

        self.HRconv = nn.Conv2d(fn, fn // 2, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(fn // 2, 1, 3, 1, 1, bias=True)

        self.refine_conv0 = nn.Conv2d(1, 64, 3, 1, 1, bias=True)
        self.refine_sas = make_layer(AltFilter, opt.angular_num, opt.layer_num_refine)
        self.refine_conv1 = nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lf_lr):
        # N=1, an2=49 (7x7)
        N, an2, H, W = lf_lr.size()

        # ==========================================
        # 1. 独立特征提取 (Feature Extraction)
        # ==========================================
        # [N*an2, 1, H, W] -> [N*an2, 64, H, W]
        lf_fea_lr = self.relu(self.fea_conv0(lf_lr.view(-1, 1, H, W)))
        lf_fea_lr = self.fea_resblock(lf_fea_lr) 

        # ==========================================
        # 2. 视点对交互 (全并行处理，无循环)
        # ==========================================
        # 这一步是为了替换原来的 for 循环。
        # 我们一次性处理所有视点对 (All Pairs)。
        # 对于 7x7，共有 49 个视点。每个视点都要和 49 个视点做 Pair。
        # 总共有 49 * 49 = 2401 个 Pair。
        
        # 定义当前处理的组大小为全部视点
        cur_size = self.an2 # 49

        # ref_fea: 参考视点特征
        # 目标: [49, 49, C, H, W] (第一维是参考视点，第二维是配对视点)
        # 解释: 第 i 个参考视点需要重复 49 次来和所有视点配对
        ref_fea = lf_fea_lr.view(cur_size, 1, -1, H, W).repeat(1, an2, 1, 1, 1)

        # view_fea: 目标视点特征
        # 目标: [49, 49, C, H, W]
        # 解释: 对于每个参考视点，我们都需要完整的 49 个视点列表
        view_fea = lf_fea_lr.view(1, an2, -1, H, W).repeat(cur_size, 1, 1, 1, 1)

        # 拼接 -> [49*49, 2C, H, W]
        # 这里 Batch Size 会变得很大 (2401)，但在导出 ONNX 结构时没问题
        pair_fea = torch.cat([ref_fea, view_fea], dim=2).view(cur_size * an2, -1, H, W)

        # === Pair Processing (一次性通过) ===
        pair_fea = self.relu(self.pair_conv0(pair_fea))
        pair_fea = self.pair_resblock(pair_fea)
        pair_fea = self.pair_conv1(pair_fea)
        
        # Reshape for Fusion
        # [49, 49, C, H, W] -> Transpose -> [49, C, 49, H, W]
        pair_fea = pair_fea.view(cur_size, an2, -1, H, W).transpose(1, 2)

        # === Fusion ===
        # 输入: [N*64, an2, H, W] -> 融合所有配对信息
        fused_fea = self.relu(self.fusion_view_conv0(pair_fea.contiguous().view(-1, an2, H, W)))
        fused_fea = self.fusion_view_resblock(fused_fea)
        fused_fea = self.relu(self.fusion_view_conv1(fused_fea)) # -> [N*49, 1, H, W]

        # Spatial Fusion
        fused_fea = self.relu(self.fusion_fea_conv0(fused_fea.view(cur_size, -1, H, W)))
        fused_fea = self.fusion_fea_resblock(fused_fea)

        # === Upsample ===
        hr_fea = self.upsampler(fused_fea)
        hr_fea = self.relu(self.HRconv(hr_fea))
        res = self.conv_last(hr_fea)
        
        # [1, 49, H*s, W*s]
        res = res.view(1, cur_size, self.scale * H, self.scale * W)

        # === Residual Base (Bicubic) ===
        # 对原始输入的所有 49 张图一次性做 Bicubic
        base = F.interpolate(lf_lr.view(-1, 1, H, W), scale_factor=self.scale, mode='bilinear', align_corners=False)
        base = base.view(1, cur_size, self.scale * H, self.scale * W)

        # 合并结果
        lf_inter = res + base

        # ==========================================
        # 3. Refine 阶段
        # ==========================================
        lf_out = self.relu(self.refine_conv0(lf_inter.view(-1, 1, self.scale * H, self.scale * W)))
        lf_out = self.refine_sas(lf_out)
        lf_out = self.refine_conv1(lf_out)
        
        # 最终恢复形状
        lf_out = lf_out.view(N, an2, self.scale * H, self.scale * W)
        lf_out += lf_inter

        return lf_out
    
      