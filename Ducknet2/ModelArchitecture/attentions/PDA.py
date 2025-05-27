import torch
import torch.nn as nn
import torch.nn.functional as F

#PDA并行注意力模块
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GlobalAvgPool(nn.Module):
    # 这里将信道减小为1
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = 0.1
        self.norm = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.Dropout3d(self.confidence_ratio)
        )
        self.channel_q = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_k = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_v = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.fc = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1,
                            padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.LN = nn.LayerNorm([in_planes,1])
    def forward(self, x):
        avg_pool = self.GAP(x)
        x_norm = self.norm(avg_pool)
        q = self.channel_q(x_norm).squeeze(-1)
        q = self.LN(q)
        k = self.channel_k(x_norm).squeeze(-1)
        k = self.LN(k)
        v = self.channel_v(x_norm).squeeze(-1)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        att = torch.matmul(alpha, v).unsqueeze(-1)
        att = att.transpose(1, 2)
        att = self.fc(att)
        att = self.sigmoid(att)

        output = (x * att) + x
        return output


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.spatial_q = nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.spatial_k = nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.spatial_v = nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
    def forward(self, x):

        q = self.spatial_q(x).squeeze(1)   # squeeze channel
        k = self.spatial_k(x).squeeze(1)
        v = self.spatial_v(x).squeeze(1)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)  # 信道权重向量:X = (SO1 2⊗α+SO1 2)

        return output

class SSEU(nn.Module):
    def __init__(self, channal, s_size=4):
        super(SSEU, self).__init__()
        self.s_size = s_size
        self.channal = channal
        self.catten = ChannelAttention(channal // s_size)
        self.satten = SpatialAttention(in_planes=channal // s_size)
        self.conv1 = BasicConv2d(channal // s_size, channal, kernel_size=1)
        self.conv1_1 = BasicConv2d(channal, channal, kernel_size=3, padding=1)

    def forward(self, x):
        # print("xSTART.shape,", x.shape)
        splits = torch.split(x, self.channal // self.s_size, dim=1)
        # print("splits.shape", splits[0].shape, splits[1].shape, splits[2].shape, splits[3].shape)
        splited = sum(splits)
        # print("splited.shape",splited.shape)
        atten = self.satten(splited) + self.catten(splited)
        atten = F.relu(self.conv1(atten))
        attens = torch.split(atten, self.channal // self.s_size, dim=1)

        a,b,c,d = (att * spl for (att, spl) in zip(attens, splits))
        # print("a.shape,", a.shape)
        # print("b.shape,", b.shape)
        # print("c.shape,", c.shape)
        # print("d.shape,", d.shape)
        out = torch.cat((a,b,c,d), dim=1)
        # print("out.shape",out.shape)
        return out + x

class pda(nn.Module):
    def __init__(self, in_channels):
        super(pda, self).__init__()
        self.sseu = SSEU(in_channels)

    def forward(self, x):
        x = self.sseu(x)
        return x