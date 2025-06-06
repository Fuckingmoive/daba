import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # print("x.shape,",x.shape)
        return out + x

class pda(nn.Module):
    def __init__(self, in_channels):
        super(pda, self).__init__()
        self.sseu = SSEU(in_channels)

    def forward(self, x):
        x = self.sseu(x)
        return x
class SSEU_e_block(nn.Module):
    def __init__(self, in_planes, out_planes, ks):
        super(SSEU_e_block, self).__init__()
        self.pd = ks // 2
        self.ks = ks
        # self.conv0 = BasicConv2d(in_planes, out_planes, kernel_size=1)
        # self.conv1xX = BasicConv2d(out_planes, out_planes, kernel_size=(1, ks), padding=(0, self.pd))
        # self.convXx1 = BasicConv2d(out_planes, out_planes, kernel_size=(ks, 1), padding=(self.pd, 0))
        # self.conv3x3 = BasicConv2d(out_planes, out_planes, kernel_size=3, padding=ks, dilation=ks)
        # 图中的SSBU
        self.sseu = SSEU(out_planes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1xX(x)
        x = self.convXx1(x)
        x = self.sseu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        return x



# class SSEU_e(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(SSEU_e, self).__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.e_block2 = SSEU_e_block(in_planes, out_planes, 3)
#         self.e_block3 = SSEU_e_block(in_planes, out_planes, 5)
#         self.e_block4 = SSEU_e_block(in_planes, out_planes, 7)
#         self.conv5_1 = BasicConv2d(3 * out_planes, 1 * out_planes, kernel_size=1)
#         self.conv5_2 = BasicConv2d(out_planes, out_planes, kernel_size=3, padding=1)
#         self.res = BasicConv2d(in_planes, out_planes, 1)
#
#     def forward(self, x):
#         x2 = F.relu(self.e_block2(x))
#         x3 = F.relu(self.e_block3(x))
#         x4 = F.relu(self.e_block4(x))
#         x0 = torch.cat((x2, x3, x4), 1)
#         x0 = self.conv5_1(x0)
#         x0 = self.conv5_2(x0)
#         x0 = F.relu(x0 + self.res(x))
#         return x0
#
#
# class SSEU_d(nn.Module):
#     def __init__(self, channel):
#         super(SSEU_d, self).__init__()
#         self.conv1 = BasicConv2d(channel * 3, channel, 1)
#         self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv5 = nn.Conv2d(channel, out_channels=1, kernel_size=3, padding=1)
#         self.sseu = SSEU(channel)
#         self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
#
#     def forward(self, f1, f2, f3):
#         f1 = self.upsample(f1, f3.shape[-2:])
#         f2 = self.upsample(f2, f3.shape[-2:])
#         f4 = torch.cat([f1, f1 * f2, f1 * f2 * f3], dim=1)
#         f4 = self.conv1(f4)
#         f4 = self.sseu(f4)
#         # print(f4.shape)
#         out = self.conv5(f4)
#         return out, f1, f2, f3, f4
#
#
# class ATT(nn.Module):
#     def __init__(self, channal):
#         super(ATT, self).__init__()
#         self.conv0 = BasicConv2d(2 * channal, channal, kernel_size=1)
#         self.conv1 = BasicConv2d(4 * channal, channal, kernel_size=1)
#         self.channal = channal
#         self.conv2 = BasicConv2d(channal, channal, kernel_size=3, padding=1)
#         self.conv3 = BasicConv2d(channal, channal, kernel_size=3, padding=1)
#         self.conv4 = BasicConv2d(channal, 1, kernel_size=1)
#
#     def forward(self, x, feature):
#         map = (x-x.min()) / (x.max()-x.min()+1e-8)
#         # trunc截断
#         x1 = (map / 0.5).trunc()
#         map = map % 0.5
#         x2 = (map / 0.25).trunc()
#         map = map % 0.25
#         x3 = (map / 0.125).trunc()
#         map = map % 0.125
#         x4 = (map / 0.0625).trunc()
#         feature = self.conv0(feature)
#         # 和四张feature map做multi
#         f1 = x1.expand(-1, self.channal, -1, -1) * feature
#         f2 = x2.expand(-1, self.channal, -1, -1) * feature
#         f3 = x3.expand(-1, self.channal, -1, -1) * feature
#         f4 = x4.expand(-1, self.channal, -1, -1) * feature
#         f = torch.cat((f1, f2, f3, f4), dim=1)
#         f = F.relu(self.conv1(f))
#         f = F.relu(self.conv2(f))
#         out = self.conv4(f)
#         out = out + x
#         return out, f
#
#
# class NPD_Net(nn.Module):
#     # res2net based encoder decoder
#     def __init__(self, channel=64, out_ch = 1):
#         super(NPD_Net, self).__init__()
#
#         # ---- ResNet Backbone ----
#         self.resnet = res2net50_v1b_26w_4s(pretrained=True)
#         # ---- SSEU_E ----
#         self.encode2_1 = SSEU_e(512, channel)
#         self.encode3_1 = SSEU_e(1024, channel)
#         self.encode4_1 = SSEU_e(2048, channel)
#         # ---- SSEU_D ----
#         self.sseu_d = SSEU_d(channel)
#         # ----- atttention-BSCA----
#         self.att4 = ATT(channel)
#         self.att3 = ATT(channel)
#         self.att2 = ATT(channel)
#
#         self.side211 = nn.Conv2d(8 * channel, 8 * channel, 1)
#         self.side233 = nn.Conv2d(64, 8 * channel, 3, padding=1)
#
#         self.side311 = nn.Conv2d(4 * channel, 4 * channel, 1)
#         self.side333 = nn.Conv2d(64, 4 * channel, 3, padding=1)
#
#         self.side411 = nn.Conv2d(2 * channel, 2 * channel, 1)
#         self.side433 = nn.Conv2d(64, 2 * channel, 3, padding=1)
#
#         self.side611 = nn.Conv2d(channel, channel, 1)
#         self.side633 = nn.Conv2d(64, channel, 3, padding=1)
#
# # -------------------------------------------------------使用的是ISNET中的loss---------------------------------------------
#
#     def compute_loss(self, preds, targets):
#         # return muti_loss_fusion(preds,targets)
#         return muti_loss_fusion(preds, targets)
# # --------------------------------------------------------------------------------------------------------------------------
#     def forward(self, x):
#
#
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)  # bs, 64, 88, 88(通道数， ， ）
#         # ---- low-level features ----
#         x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
#         x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
#         x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
#         x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
#
#         x2_encode = self.encode2_1(x2)  # bs, 64, 44, 44
#         x3_encode = self.encode3_1(x3)  # bs, 64, 22, 22
#         x4_encode = self.encode4_1(x4)  # bs, 64, 11, 11
#
#         crop_5, x4, x3, x2, f5 = self.sseu_d(x4_encode, x3_encode, x2_encode)  # bs 64 44 44
#         # --------------------------------------------------------
#         x4 = self.side611(self.side633(x4))
#         x33 = self.side411(self.side433(x3))
#         x335 = self.side311(self.side333(x2))
#         x22 = self.side211(self.side233(f5))
#
#         # ---------------------------------------------------------
#
#         lateral_map_5 = F.interpolate(crop_5, scale_factor=8,
#                                       mode='bilinear')  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#
#         crop_4, f4 = self.att4(crop_5, torch.cat((x4, f5), dim=1))
#         lateral_map_4 = F.interpolate(crop_4, scale_factor=8,
#                                       mode='bilinear')  # NOTES: Sup-2 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#
#         crop_3, f3 = self.att3(crop_4, torch.cat((x3, f4), dim=1))
#         lateral_map_3 = F.interpolate(crop_3, scale_factor=8,
#                                       mode='bilinear')  # NOTES: Sup-3 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#
#         crop_2, f2 = self.att2(crop_3, torch.cat((x2, f3), dim=1))
#         lateral_map_2 = F.interpolate(crop_2, scale_factor=8,
#                                       mode='bilinear')  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
#
#
#         return [lateral_map_5, lateral_map_5,  lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_2], \
#                [x4, x4, x33, x335, x22, x22]
#
# ## upsample tensor 'src' to have the same spatial size with tensor 'tar'
# def _upsample_like(src,tar):
#     src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
#     return src
#
#
# # ---------------------------------------------------------------------------
#
# fea_loss = nn.MSELoss(size_average=True)
# kl_loss = nn.KLDivLoss(size_average=True)
# l1_loss = nn.L1Loss(size_average=True)
# smooth_l1_loss = nn.SmoothL1Loss(size_average=True)
# bce_loss = nn.BCELoss(size_average=True)  # （Binary CrossEntropy Loss）
#
# def muti_loss_fusion(preds, target):
#     loss0 = 0.0
#     loss = 0.0
#
#     for i in range(0,len(preds)):
#         if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
#             tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
#             loss = loss + bce_loss(preds[i], tmp_target)
#         else:
#             loss = loss + bce_loss(preds[i], target)
#         if(i==0):
#             loss0 = loss
#     return loss0, loss
# -------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------
# if __name__ == '__main__':
#     ras = NPD_Net().cuda()
#     input_tensor = torch.randn(1, 3, 352, 352).cuda()
#
#     out = ras(input_tensor)