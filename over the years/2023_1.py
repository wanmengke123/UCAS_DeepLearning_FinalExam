import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch中的神经网络模块
import torch.nn.functional as F  # 导入PyTorch中的函数式API
from torchvision import models  # 导入torchvision中的模型模块

# 空间注意力模块定义
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 1x1卷积层，用于生成注意力图
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Sigmoid激活函数，用于将注意力图归一化到0-1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算注意力图
        attention = self.sigmoid(self.conv1(x))
        # 将输入特征图与注意力图相乘，调整特征图的权重
        return x * attention

# 多任务损失函数定义
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, edge_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        # 分割损失和边缘检测损失的权重
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight
        # 分割任务的交叉熵损失函数
        self.seg_loss = nn.CrossEntropyLoss()
        # 边缘检测任务的二分类交叉熵损失函数
        self.edge_loss = nn.BCEWithLogitsLoss()

    def forward(self, seg_output, seg_target, edge_output, edge_target):
        # 计算分割损失
        seg_loss = self.seg_loss(seg_output, seg_target)
        # 计算边缘检测损失
        edge_loss = self.edge_loss(edge_output, edge_target)
        # 返回加权后的总损失
        return self.seg_weight * seg_loss + self.edge_weight * edge_loss

# 包含注意力机制的U-Net模型定义
class UNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(UNetWithAttention, self).__init__()
        # 使用预训练的ResNet34作为编码器
        self.encoder = models.resnet34(pretrained=True)
        # 空间注意力模块
        self.sa = SpatialAttention(512)

        # 解码器部分，通过上采样逐步恢复分割结果
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # 上采样并卷积
            nn.ReLU(inplace=True), # ReLU激活
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 上采样并卷积
            nn.ReLU(inplace=True), # ReLU激活
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 上采样并卷积
            nn.ReLU(inplace=True), # ReLU激活
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 上采样并卷积
            nn.ReLU(inplace=True), # ReLU激活
            nn.Conv2d(32, num_classes, kernel_size=1) # 最后一层卷积，用于生成分割结果
        )
        # 边缘检测头，用于边缘检测任务
        self.edge_head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # 空间注意力机制
        x = self.sa(x)

        # 边缘检测输出
        edge_output = self.edge_head(x)

        # 解码器部分，恢复分割结果
        x = self.decoder(x)

        # 返回分割结果和边缘检测结果
        return x, edge_output

# 示例用法
if __name__ == "__main__":
    # 创建模型实例，Pascal VOC有21类
    model = UNetWithAttention(num_classes=21)
    # 创建多任务损失函数实例
    criterion = MultiTaskLoss()

    # 创建模拟输入图像和标签
    input_image = torch.randn(1, 3, 224, 224)  # 模拟输入图像
    seg_target = torch.randint(0, 21, (1, 224, 224))  # 模拟分割标签
    edge_target = torch.randint(0, 2, (1, 1, 224, 224)).float()  # 模拟边缘标签

    # 前向传播，获取分割结果和边缘检测结果
    seg_output, edge_output = model(input_image)
    # 计算损失
    loss = criterion(seg_output, seg_target, edge_output, edge_target)

    # 打印损失值
    print("Loss:", loss.item())
