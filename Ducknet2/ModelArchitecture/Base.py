import torch
import torch.nn as nn
import torch.nn.functional as F

from Ducknet2.CustomLayers.ConvBlock2D_Test2 import ConvBlock2D
# from AFM.MSAA import MSAA as atten
# from Ducknet2.ModelArchitecture.attentions.LSKA import Attention as atten
from Ducknet2.ModelArchitecture.lskatt.LSKAtest12 import Attention as atten
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
        # self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        # self.t0_re = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        # self.t0_re = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)
        self.t0_re = nn.Conv2d(starting_filters, starting_filters, kernel_size=2, stride=2, padding=0)

        # 连接和块
        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.t1_block = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        # self.t1_reblock = ConvBlock2D(starting_filters * 2, starting_filters * 1, 'duckv2', repeat=1)
        self.t1_reblock = nn.Conv2d(starting_filters * 2, starting_filters * 1, kernel_size=2, stride=2, padding=0)

        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.t2_block = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        # self.t2_reblock = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)
        self.t2_reblock = nn.Conv2d(starting_filters * 4, starting_filters * 2, kernel_size=2, stride=2, padding=0)

        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.t3_block = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        # self.t3_reblock = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)
        self.t3_reblock = nn.Conv2d(starting_filters * 8, starting_filters * 4, kernel_size=2, stride=2, padding=0)

        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.t4_block = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)
        # self.t4_reblock = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)
        self.t4_reblock = nn.Conv2d(starting_filters * 16, starting_filters * 8, kernel_size=2, stride=2, padding=0)

        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
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
        # self.att0 = atten(starting_filters)
        # self.att1 = atten(starting_filters * 2)
        # self.att2 = atten(starting_filters * 4)
        # self.att3 = atten(starting_filters * 8)
        # self.att4 = atten(starting_filters * 16)
        # self.att0 = atten(starting_filters, k_size=35)
        # self.att1 = atten(starting_filters * 2, k_size=35)
        # self.att2 = atten(starting_filters * 4, k_size=35)
        # self.att3 = atten(starting_filters * 8, k_size=35)
        # self.att4 = atten(starting_filters * 16, k_size=35)
        # self.att0 = atten(starting_filters,starting_filters)
        # self.att1 = atten(starting_filters * 2,starting_filters * 2)
        # self.att2 = atten(starting_filters * 4,starting_filters * 4)
        # self.att3 = atten(starting_filters * 8,starting_filters * 8)
        # self.att4 = atten(starting_filters * 16,starting_filters * 16)

        # 最终卷积层
        self.final_conv = nn.Conv2d(starting_filters, out_classes, kernel_size=1)

    def forward(self, x):
        #print('Starting DUCK-Net')
        # print("X.size():", x.size())
        p1 = self.p1(x)
        # print("p1.size():", p1.size())
        p2 = self.p2(p1)
        # print("p2.size():", p2.size())
        p3 = self.p3(p2)
        # print("p3.size():", p3.size())
        p4 = self.p4(p3)
        # print("p4.size():", p4.size())
        p5 = self.p5(p4)
        # print("p5.size():", p5.size())

        t0 = self.t0(x)
        # print("t0.size():", t0.size())
        # t0 = self.att0(t0)  # 注意力机制
        # print("t0.size():", t0.size())

        l1i = self.l1i(t0)
        # print("l1i.size():", l1i.size())
        s1 = l1i + p1
        # print("s1.size():", s1.size())
        # t1 = self.t1_block(s1)
        t1 = p1
        # print("t1.size():", t1.size())
        # t1 = self.att1(t1)  # 注意力机制
        # print("t1.size():", t1.size())

        l2i = self.l2i(t1)
        # print("l2i.size():", l2i.size())
        s2 = l2i + p2
        # print("s2.size():", s2.size())
        # t2 = self.t2_block(s2)
        t2 = p2
        # t2 = self.att2(t2)  # 注意力机制

        l3i = self.l3i(t2)
        # print("l3i.size():", l3i.size())
        s3 = l3i + p3
        # print("s3.size():", s3.size())
        # t3 = self.t3_block(s3)
        t3 = p3
        # print("t3.size():", t3.size())
        # t3 = self.att3(t3)  # 注意力机制

        l4i = self.l4i(t3)
        # print("l4i.size():", l4i.size())
        s4 = l4i + p4
        # print("s4.size():", s4.size())
        # t4 = self.t4_block(s4)
        t4 = p4
        # print("t4.size():", t4.size()q1
        # t4 = self.att4(t4)  # 注意力机制

        l5i = self.l5i(t4)
        # print("l5i.size():", l5i.size())
        # s5 = l5i + p5
        s5 = p5
        # print("s5.size():", s5.size())
        t51 = self.t51(s5)
        t53 = self.t53(t51)
        # print("t53.size():", t53.size())

        # 上采样路径
        l5o = F.interpolate(t53, scale_factor=2, mode='nearest')
        # print("l5o.size():", l5o.size())
        c4 = l5o + t4
        c4 = t4
        # print("c4.size():", c4.size())
        c4 = self.groupconv4(c4)
        q4 = self.t4_reblock(c4)
        # print("q4.size():", q4.size())

        l4o = F.interpolate(q4, scale_factor=4, mode='nearest')
        # print("l4o.size():", l4o.size())
        c3 = l4o + t3
        # print("c3.size():", c3.size())
        c3 = self.groupconv3(c3)
        q3 = self.t3_reblock(c3)
        # print("q3.size():", q3.size())


        l3o = F.interpolate(q3, scale_factor=4, mode='nearest')
        # print("l3o.size():", l3o.size())
        c2 = l3o + t2
        # print("c2.size():", c2.size())
        c2 = self.groupconv2(c2)
        q2 = self.t2_reblock(c2)
        # print("q2.size():", q2.size())

        l2o = F.interpolate(q2, scale_factor=4, mode='nearest')
        # print("l2o.size():", l2o.size())
        c1 = l2o + t1
        # print("c1.size():", c1.size())q1
        c1 = self.groupconv1(c1)
        q1 = self.t1_reblock(c1)
        # print("q1.size():", q1.size())

        l1o = F.interpolate(q1, scale_factor=4, mode='nearest')
        # print("l1o.size():", l1o.size())
        c0 = l1o + t0
        # print("c0.size():", c0.size())
        c0 = self.groupconv0(c0)
        z1 = self.t0_re(c0)
        # print("z1.size():", z1.size())

        output = self.final_conv(z1)
        output= F.interpolate(output, scale_factor=2, mode='nearest')
        # print("output.size():", output.size())
        # print("out.size():", torch.sigmoid(output).size())
        return torch.sigmoid(output)

# 示例：创建模型
# model = DuckNet(img_height=256, img_width=256, input_channels=3, out_classes=1, starting_filters=16)
