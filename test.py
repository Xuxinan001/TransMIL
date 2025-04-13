import torch
import torch.nn as nn

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
# 创建PPEG实例
ppeg = PPEG(dim=512)

# 创建模拟输入
B, C = 2, 512
H, W = 4, 4
x = torch.randn(B, 1+H*W, C)  # 分类token + 特征token

# 前向传播
output = ppeg(x, H, W)

# 检查输入输出形状
print("Input shape:", x.shape)      # torch.Size([2, 17, 512])
print("Output shape:", output.shape) # torch.Size([2, 17, 512])
print("ss")