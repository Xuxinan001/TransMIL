import torch
import torch.nn as nn

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        # 卷积
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        print("x.shape:",x.shape)
        # 分类token(cls_token)和特征token(feat_token)
        cls_token, feat_token = x[:, 0], x[:, 1:]
        # cls.shape: torch.Size([2, 512])
        # feat.shape: torch.Size([2, 16, 512])
        print("cls.shape:",cls_token.shape)
        print("feat.shape:",feat_token.shape)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        # 只是相加，维度没变
        print("x.shape:",x.shape)
        # flatten展平，对于这段代码，是将第3个维度之后都合并成1维
        x = x.flatten(2).transpose(1, 2)
        print("x.shape:",x.shape)
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