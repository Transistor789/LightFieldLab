import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# =========================================================================
# 1. 补充缺失的工具函数 (使用原生 PyTorch 实现，替换 einops)
# =========================================================================


def LF_interpolate(LF, scale_factor, mode='bicubic'):
    # 输入: [B, C, U, V, H, W]
    b, c, u, v, h, w = LF.shape
    # 压扁: [B*U*V, C, H, W]
    LF = LF.view(b * u * v, c, h, w)
    
    # 插值
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    
    # 恢复: [B, C, U, V, H*scale, W*scale]
    h_new, w_new = LF_upscale.shape[2], LF_upscale.shape[3]
    LF_upscale = LF_upscale.view(b, c, u, v, h_new, w_new)
    return LF_upscale

# =========================================================================
# 2. 修改后的网络结构 (去 einops + 向量化 Mask)
# =========================================================================

class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)
        self.mask_field = [0, 0] # 占位符

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int, device):
        # 【关键修改】完全向量化的 Mask 生成，无循环，支持动态尺寸
        
        # 1. 生成坐标网格
        # idx_h: [0, 0, ..., 1, 1, ...] (重复 w 次)
        idx_h = torch.arange(h, device=device).view(-1, 1).expand(h, w).reshape(-1)
        # idx_w: [0, 1, ..., 0, 1, ...] (重复 h 次)
        idx_w = torch.arange(w, device=device).view(1, -1).expand(h, w).reshape(-1)

        # 2. 广播计算两两距离 [HW, HW]
        # diff_h[i, j] = idx_h[i] - idx_h[j]
        diff_h = idx_h.view(-1, 1) - idx_h.view(1, -1)
        diff_w = idx_w.view(-1, 1) - idx_w.view(1, -1)

        # 3. 窗口逻辑判断 (Window Logic)
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left

        # 掩码条件: -left <= diff < right
        mask_h = (diff_h >= -k_h_left) & (diff_h < k_h_right)
        mask_w = (diff_w >= -k_w_left) & (diff_w < k_w_right)
        mask = mask_h & mask_w

        # 4. 转为 float mask (-inf, 0)
        # MultiheadAttention 中: 0 表示保留，-inf 表示遮蔽
        float_mask = torch.zeros_like(mask, dtype=torch.float)
        float_mask.masked_fill_(~mask, float('-inf'))

        return float_mask

    def forward(self, buffer):
        # buffer input: [B, C, N, V, W]
        # 这里的 V, W 对应 AltFilter 里的 (AngRes, H) 或 (AngRes, W)
        b, c, n, v, w = buffer.shape
        
        # 动态生成 Mask
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1], buffer.device)

        # 维度变换: [B, C, N, V, W] -> [V*W, B*N, C] (Sequence, Batch, Feature)
        # 1. Permute -> [V, W, B, N, C]
        epi_token = buffer.permute(3, 4, 0, 2, 1).contiguous()
        # 2. View -> [V*W, B*N, C]
        epi_token = epi_token.view(v * w, b * n, c)

        epi_token = self.linear_in(epi_token)
        epi_token_norm = self.norm(epi_token)
        
        # Self-Attention
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        
        # 维度恢复: [V*W, B*N, C] -> [B, C, N, V, W]
        # 1. View -> [V, W, B, N, C]
        buffer = epi_token.view(v, w, b, n, c)
        # 2. Permute -> [B, C, N, V, W]
        buffer = buffer.permute(2, 4, 3, 0, 1).contiguous()

        return buffer

class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        shortcut = buffer
        # buffer: [B, C, U*V, H, W]
        b, c, uv, h, w = buffer.size()
        u = self.angRes
        v = self.angRes
        
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # ---------------- Horizontal Branch ----------------
        # 原始: rearrange(buffer, 'b c (u v) h w -> b c (v w) u h')
        # 拆解: [B, C, U, V, H, W] -> [B, C, V, W, U, H] -> [B, C, V*W, U, H]
        
        # 1. View 拆分 U, V
        temp = buffer.view(b, c, u, v, h, w)
        # 2. Permute: [B, C, V, W, U, H] (indices: 0, 1, 3, 5, 2, 4)
        temp = temp.permute(0, 1, 3, 5, 2, 4).contiguous()
        # 3. View 合并
        temp = temp.view(b, c, v*w, u, h)
        
        temp = self.epi_trans(temp)
        
        # 恢复: rearrange(..., 'b c (v w) u h -> b c (u v) h w')
        # [B, C, V*W, U, H] -> [B, C, V, W, U, H] -> [B, C, U, V, H, W] -> [B, C, U*V, H, W]
        temp = temp.view(b, c, v, w, u, h)
        temp = temp.permute(0, 1, 4, 2, 5, 3).contiguous() # -> [B, C, U, V, H, W]
        temp = temp.view(b, c, u*v, h, w)
        
        buffer = self.conv(temp) + shortcut

        # ---------------- Vertical Branch ----------------
        # 原始: rearrange(buffer, 'b c (u v) h w -> b c (u h) v w')
        # 拆解: [B, C, U, V, H, W] -> [B, C, U, H, V, W] -> [B, C, U*H, V, W]
        
        # 1. View 拆分
        temp = buffer.view(b, c, u, v, h, w)
        # 2. Permute: [B, C, U, H, V, W] (indices: 0, 1, 2, 4, 3, 5)
        temp = temp.permute(0, 1, 2, 4, 3, 5).contiguous()
        # 3. View 合并
        temp = temp.view(b, c, u*h, v, w)
        
        temp = self.epi_trans(temp)
        
        # 恢复
        temp = temp.view(b, c, u, h, v, w)
        temp = temp.permute(0, 1, 2, 4, 3, 5).contiguous() # -> [B, C, U, V, H, W]
        temp = temp.view(b, c, u*v, h, w)
        
        buffer = self.conv(temp) + shortcut

        return buffer

class get_model(nn.Module):
    def __init__(self, angRes, scale):
        super(get_model, self).__init__()
        channels = 64
        self.angRes = angRes
        self.scale = scale

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
        )

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr):
        # ==========================================================
        # 【修改点 1】: 移除 RGB -> YCbCr
        # 假设输入 lr 已经是单通道 Y [B, 1, U, V, H, W]
        # ==========================================================
        
        b, c, u, v, h, w = lr.size()

        # 1. 直接对输入做 Bicubic 上采样作为残差基准 (Base)
        # 输入是 Y，输出也是 Y
        sr_y_bicubic = LF_interpolate(lr, scale_factor=self.scale, mode='bicubic')

        # 2. 特征提取
        # 不需要再切片 lr_ycbcr[:, 0:1]，因为输入本身就是 1 通道
        x = lr.view(b, 1, u*v, h, w)
        
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        # 3. 网络推理
        buffer = self.altblock(buffer) + buffer

        # 4. 上采样
        buffer = buffer.view(b, 64, u, v, h, w)
        buffer = buffer.permute(0, 1, 2, 4, 3, 5).contiguous()
        buffer = buffer.view(b, 64, u*h, v*w)
        
        y_residual = self.upsampling(buffer)
        
        # 维度恢复
        h_new = y_residual.shape[2] // u
        w_new = y_residual.shape[3] // v
        y_residual = y_residual.view(b, 1, u, h_new, v, w_new)
        y_residual = y_residual.permute(0, 1, 2, 4, 3, 5).contiguous()

        # ==========================================================
        # 【修改点 2】: 移除 YCbCr -> RGB
        # 直接输出 SR_Y = Bicubic_Y + Network_Residual
        # ==========================================================
        out = sr_y_bicubic + y_residual
        
        return out
