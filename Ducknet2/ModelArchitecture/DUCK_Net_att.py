import torch
import torch.nn as nn
import torch.nn.functional as F

from Ducknet2.CustomLayers.ConvBlock2D_Test import ConvBlock2D
from Ducknet2.ModelArchitecture.attentions.SCSE import SCSEModule as atten
#在跳跃连接中加入模块

class DuckNet(nn.Module):
    def __init__(self, img_height, img_width, input_channels, out_classes, starting_filters):
        super(DuckNet, self).__init__()

        # 下采样路径
        self.p1 = nn.Conv2d(input_channels, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.p2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.p3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.p5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)

        # 跳跃连接的初始卷积块
        self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        self.t0_re = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        # 连接和块
        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.t1_block = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.t1_reblock = ConvBlock2D(starting_filters * 2, starting_filters * 1, 'duckv2', repeat=1)

        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.t2_block = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.t2_reblock = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)

        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.t3_block = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.t3_reblock = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)

        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.t4_block = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)
        self.t4_reblock = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)

        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        #self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t51 = nn.Sequential(
            ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=1),
            ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=1)
        )
        self.t53 = nn.Sequential(
            ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=1),
            ConvBlock2D(starting_filters * 16, starting_filters * 16, 'resnet', repeat=1)
        )

        # 群巻积
        self.groupconv0 = nn.Conv2d(starting_filters, starting_filters, kernel_size=1, stride=1, padding=0, groups=starting_filters)
        self.groupconv1 = nn.Conv2d(starting_filters * 2, starting_filters * 2, kernel_size=1, stride=1, padding=0, groups=starting_filters * 2)
        self.groupconv2 = nn.Conv2d(starting_filters * 4, starting_filters * 4, kernel_size=1, stride=1, padding=0, groups=starting_filters * 4)
        self.groupconv3 = nn.Conv2d(starting_filters * 8, starting_filters * 8, kernel_size=1, stride=1, padding=0, groups=starting_filters * 8)
        self.groupconv4 = nn.Conv2d(starting_filters * 16, starting_filters * 16, kernel_size=1, stride=1, padding=0, groups=starting_filters * 16)

        # 跳跃连接的注意力机制
        self.att0 = atten(starting_filters)
        self.att1 = atten(starting_filters * 2)
        self.att2 = atten(starting_filters * 4)
        self.att3 = atten(starting_filters * 8)
        self.att4 = atten(starting_filters * 16)

        # 最终卷积层
        self.final_conv = nn.Conv2d(starting_filters, out_classes, kernel_size=1)

    def forward(self, x):
        #print('Starting DUCK-Net')

        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0 = self.t0(x)
        # t0 = self.att0(t0)  # 注意力机制

        l1i = self.l1i(t0)
        s1 = l1i + p1
        t1 = self.t1_block(s1)
        # t1 = self.att1(t1)  # 注意力机制

        l2i = self.l2i(t1)
        s2 = l2i + p2
        t2 = self.t2_block(s2)
        # t2 = self.att2(t2)  # 注意力机制

        l3i = self.l3i(t2)
        s3 = l3i + p3
        t3 = self.t3_block(s3)
        # t3 = self.att3(t3)  # 注意力机制

        l4i = self.l4i(t3)
        s4 = l4i + p4
        t4 = self.t4_block(s4)
        # t4 = self.att4(t4)  # 注意力机制

        l5i = self.l5i(t4)
        s5 = l5i + p5
        t51 = self.t51(s5)
        t53 = self.t53(t51)

        # 上采样路径
        l5o = F.interpolate(t53, scale_factor=2, mode='nearest')
        c4 = l5o + t4
        c4 = self.att4(c4)  # 注意力机制
        c4 = self.groupconv4(c4)
        q4 = self.t4_reblock(c4)

        l4o = F.interpolate(q4, scale_factor=2, mode='nearest')
        c3 = l4o + t3
        c3 = self.att3(c3)  # 注意力机制
        c3 = self.groupconv3(c3)
        q3 = self.t3_reblock(c3)


        l3o = F.interpolate(q3, scale_factor=2, mode='nearest')
        c2 = l3o + t2
        c2 = self.att2(c2)  # 注意力机制
        c2 = self.groupconv2(c2)
        q2 = self.t2_reblock(c2)

        l2o = F.interpolate(q2, scale_factor=2, mode='nearest')
        c1 = l2o + t1
        c1 = self.att1(c1)  # 注意力机制
        c1 = self.groupconv1(c1)
        q1 = self.t1_reblock(c1)

        l1o = F.interpolate(q1, scale_factor=2, mode='nearest')
        c0 = l1o + t0
        c0 = self.att0(c0)  # 注意力机制
        c0 = self.groupconv0(c0)
        z1 = self.t0_re(c0)

        output = self.final_conv(z1)
        return torch.sigmoid(output)

# 示例：创建模型
# model = DuckNet(img_height=256, img_width=256, input_channels=3, out_classes=1, starting_filters=16)
