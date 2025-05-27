import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_initializer = 'he_uniform'


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
        super(ConvBlock2D, self).__init__()
        self.layers = []
        for _ in range(0, repeat):
            if block_type == 'separated':
                self.layers.append(SeparatedConv2DBlock(in_channels, filters, size=size, padding=1))
            elif block_type == 'duckv2':
                self.layers.append(Duckv2Conv2DBlock(in_channels, filters, size=size))
            elif block_type == 'midscope':
                self.layers.append(MidScopeConv2DBlock(in_channels, filters))
            elif block_type == 'widescope':
                self.layers.append(WideScopeConv2DBlock(in_channels, filters))
            elif block_type == 'resnet':
                self.layers.append(ResNetConv2DBlock(in_channels, filters, dilation_rate))
            elif block_type == 'conv':
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size=size, padding=1))
                self.layers.append(nn.ReLU(inplace=True))
                in_channels = filters  # Update channels for next layer
            elif block_type == 'double_convolution':
                self.layers.append(DoubleConvolutionWithBatchNormalization(in_channels, filters, dilation_rate))
            elif block_type == 'eleven':
                self.layers.append(ElevenConv2DBlock(in_channels, filters))
            elif block_type == 'seventeen':
                self.layers.append(SeventeenConv2DBlock(in_channels, filters))
            else:
                raise ValueError("Invalid block type")

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#初始并联结构
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.wide_scope(x)
#
#         x2 = self.mid_scope(x)
#
#         x3 = self.resnet1(x)
#
#         x4 = self.resnet2(x)
#
#         x5 = self.resnet3(x)
#
#         x6 = self.separated(x)
#
#         x7 = x1 + x2 + x3 + x4 + x5 + x6
#
#         x7 = self.batch_norm2(x7)  # Final batch normalization
#
#         return x7

# #将并联改为串联，222,巻积内核由小到大
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通import torch
import torch.nn as nn

# 假设输入特征图的通道数为8
in_channels = 8

# 定义一个卷积层，kernel_size为2，填充为1，步幅为1，输出通道数为输入通道数的一半
conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=1, padding=1)

# 创建一个示例输入特征图，假设初始大小为8通道，4x4的空间维度
input_feature_map = torch.randn(1, in_channels, 4, 4)

# 应用卷积操作
output_feature_map = conv(input_feature_map)

# print(f'输入特征图大小: {input_feature_map.shape}')
# print(f'输出特征图大小: {output_feature_map.shape}')
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)
#         x2 = self.mid_scope(x)
#         x = x1 + x2
#         x = self.batch_norm2(x)
#
#         x3 = self.resnet2(x)
#         x4 = self.separated(x)
#         x = x3 + x4
#         x = self.batch_norm2(x)
#
#         x5 = self.resnet3(x)
#         x6 = self.wide_scope(x)
#         x7 = x5 + x6import torch
import torch.nn as nn

# 假设输入特征图的通道数为8
in_channels = 8

# 定义一个卷积层，kernel_size为2，填充为1，步幅为1，输出通道数为输入通道数的一半
conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=1, padding=1)

# 创建一个示例输入特征图，假设初始大小为8通道，4x4的空间维度
input_feature_map = torch.randn(1, in_channels, 4, 4)

# 应用卷积操作
output_feature_map = conv(input_feature_map)

# print(f'输入特征图大小: {input_feature_map.shape}')
# print(f'输出特征图大小: {output_feature_map.shape}')
#         x7 = self.batch_norm2(x7)  # Final batch normalization
#
#         return x7

# #将并联改为串联，222,使用lska模块，巻积内核由小到大
class Duckv2Conv2DBlock(nn.Module):
    def __init__(self, in_channels, filters, size):
        super(Duckv2Conv2DBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
        # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
        self.wide_scope = WideScopeConv2DBlock(filters, filters)
        self.mid_scope = MidScopeConv2DBlock(filters, filters)
        self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
        self.resnet2 = nn.Sequential(
            ResNetConv2DBlock(filters, filters, dilation_rate=1),
            ResNetConv2DBlock(filters, filters, dilation_rate=1)
        )
        self.resnet3 = nn.Sequential(
            ResNetConv2DBlock(filters, filters, dilation_rate=1),
            ResNetConv2DBlock(filters, filters, dilation_rate=1),
            ResNetConv2DBlock(filters, filters, dilation_rate=1),
        )
        self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
        self.lska1 = LSKA7(filters, filters)  #7
        self.lska2 = LSKA11(filters, filters)  # 11
        self.batch_norm2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.initial_conv(x)  # 调整通道数到 filters

        x1 = self.resnet1(x)  # 5
        x = x1
        x = self.batch_norm2(x)

        x2 = self.lska1(x)  # 7
        x3 = self.mid_scope(x)  # 7
        x = x2 + x3
        x = self.batch_norm2(x)

        x4 = self.resnet2(x)  # 9
        x5 = self.lska2(x)  # 11
        x6 = self.wide_scope(x)  # 13
        x7 = x4 + x5 + x6
        x7 = self.batch_norm2(x7)  # Final batch normalization

        return x7


# #将并联改为串联，222,使用lska模块，巻积内核由小到大
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.lska1 = LSKA7(filters, filters)#7
#         self.lska2 = LSKA11(filters, filters) # 11
#         self.lska3 = LSKA15(filters, filters) # 15
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)import torch
import torch.nn as nn

# 假设输入特征图的通道数为8
in_channels = 8

# 定义一个卷积层，kernel_size为2，填充为1，步幅为1，输出通道数为输入通道数的一半
conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=1, padding=1)

# 创建一个示例输入特征图，假设初始大小为8通道，4x4的空间维度
input_feature_map = torch.randn(1, in_channels, 4, 4)

# 应用卷积操作
output_feature_map = conv(input_feature_map)

# print(f'输入特征图大小: {input_feature_map.shape}')
# print(f'输出特征图大小: {output_feature_map.shape}')
#         x = self.batch_norm2(x)
#
#         x3 = self.resnet2(x)#9
#         x4 = self.lska2(x)#11
#         x = x3 + x4
#         x = self.batch_norm2(x)
#
#         x5 = self.wide_scope(x)#13
#         x6 = self.lska3(x)  # 15
#         x7 = x5 + x6
#         x7 = self.batch_norm2(x7)  # Final batch normalization
#
#         return x7


#将并联改为串联，2222,使用lska模块，巻积内核由小到大
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.lska1 = LSKA7(filters,filters)#7
#         self.lska2 = LSKA11(filters, filters) # 11
#         self.lska3 = LSKA23(filters, filters) # 23
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)#5
#         x2 = self.lska1(x)#7
#         x = x1 + x2
#         x = self.batch_norm2(x)
#
#         x3 = self.separated(x)  # 7
#         x4 = self.mid_scope(x)  # 7
#         x = x3 + x4
#         x = self.batch_norm2(x)
#
#         x5 = self.resnet2(x)#9
#         x6 = self.lska2(x)#11
#         x = x5 + x6
#         x = self.batch_norm2(x)
#
#         x7 = self.wide_scope(x)#13
#         x8 = self.lska3(x)#23
#         x = x7 + x8
#         x = self.batch_norm2(x)  # Final batch normalization
#
#         return x

#将并联改为串联，322,使用lska模块，巻积内核由小到大
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.lska1 = LSKA7(filters,filters)#7
#         self.lska2 = LSKA11(filters, filters) # 11
#         self.lska3 = LSKA23(filters, filters) # 23
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)#5
#         x2 = self.lska1(x)#7
#         x3 = self.mid_scope(x)#7
#         x = x1 + x2 + x3
#         x = self.batch_norm2(x)
#
#         x4 = self.resnet2(x)#9
#         x5 = self.lska2(x)#11
#         x = x4 + x5
#         x = self.batch_norm2(x)
#
#         x6 = self.wide_scope(x)#13
#         x7 = self.lska3(x)#23
#         x = x6 + x7
#         x = self.batch_norm2(x)  # Final batch normalization
#
#         return x

#将并联改为串联，222,巻积内核由小到大,57791315
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.eleven = ElevenConv2DBlock(filters, filters)
#         self.fifteen = FifteenConv2DBlock(filters, filters)
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)
#         x2 = self.mid_scope(x)
#         x = x1 + x2
#         x = self.batch_norm2(x)
#
#         x3 = self.resnet2(x)
#         x4 = self.separated(x)
#         x = x3 + x4
#         x = self.batch_norm2(x)
#
#         x5 = self.resnet3(x)
#         x6 = self.fifteen(x)
#         x7 = x5 + x6
#         x7 = self.batch_norm2(x7)  # Final batch normalization
#
#         return x7

#将并联改为串联33,57791313
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)
#         x2 = self.mid_scope(x)
#         x3 = self.separated(x)
#         x = x1 + x2 + x3
#         x = self.batch_norm2(x)
#
#         x4 = self.resnet2(x)
#         x5 = self.resnet3(x)
#         x6 = self.wide_scope(x)
#         x = x4 + x5 + x6
#         x7 = self.batch_norm2(x)  # Final batch normalization
#
#         return x7

#将并联改为串联，322,5779111313
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.eleven = ElevenConv2DBlock(filters, filters)  # 11
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)
#         x2 = self.mid_scope(x)
#         x3 = self.separated(x)
#         x = x1 + x2 + x3
#         x = self.batch_norm2(x)
#
#         x4 = self.resnet2(x)
#         x5 = self.eleven(x)
#         x = x4 + x5
#         x = self.batch_norm2(x)
#
#         x6 = self.resnet3(x)
#         x7 = self.wide_scope(x)
#         x = x6 + x7
#         x8 = self.batch_norm2(x)  # Final batch normalization
#
#         return x8

#将并联改为串联，232,5779111313
# class Duckv2Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size):
#         super(Duckv2Conv2DBlock, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels)
#         self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=1)  # 调整通道数
#         # self.initial_conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 调整通道数
#         self.wide_scope = WideScopeConv2DBlock(filters, filters)
#         self.mid_scope = MidScopeConv2DBlock(filters, filters)
#         self.eleven = ElevenConv2DBlock(filters, filters)  # 11
#         self.resnet1 = ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         self.resnet2 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1)
#         )
#         self.resnet3 = nn.Sequential(
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#             ResNetConv2DBlock(filters, filters, dilation_rate=1),
#         )
#         self.separated = SeparatedConv2DBlock(filters, filters, size=7, padding=3)
#         self.batch_norm2 = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.initial_conv(x)  # 调整通道数到 filters
#
#         x1 = self.resnet1(x)
#         x2 = self.mid_scope(x)
#         x = x1 + x2
#         x = self.batch_norm2(x)
#
#         x3 = self.separated(x)
#         x4 = self.resnet2(x)
#         x5 = self.eleven(x)
#         x = x3 + x4 + x5
#         x = self.batch_norm2(x)
#
#         x6 = self.resnet3(x)
#         x7 = self.wide_scope(x)
#         x = x6 + x7
#         x8 = self.batch_norm2(x)  # Final batch normalization
#
#         return x8


# class SeparatedConv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, size=3, padding=1):
#         super(SeparatedConv2DBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, filters, kernel_size=(1, size), padding=(0, padding)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filters, filters, kernel_size=(size, 1), padding=(padding, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.block(x)

#7*7
class LSKA7(nn.Module):
    def __init__(self, in_channels, dim):
        super(LSKA7, self).__init__()

        self.conv0h = nn.Conv2d(in_channels, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2),
                                groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                        dilation=2)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                        dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0h(x)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv0v(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv_spatial_h(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv_spatial_v(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv1(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        y = u * attn
        y = self.relu(y)
        y = self.bn(y)
        return y


#7*7
class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, padding=1):
        super(SeparatedConv2DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, size), padding=(0, padding), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(size, 1), padding=(padding, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


#7*7
class MidScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(MidScopeConv2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.block(x)


#11*11
class LSKA11(nn.Module):
    def __init__(self, in_channel, dim):
        super(LSKA11, self).__init__()

        self.conv0h = nn.Conv2d(in_channel, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2),
                                groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                        dilation=2)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                        dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0h(x)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv0v(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv_spatial_h(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv_spatial_v(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv1(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        y = u * attn
        y = self.relu(y)
        y = self.bn(y)

        return y


#15*15
class LSKA15(nn.Module):
    def __init__(self, in_channel, dim):
        super(LSKA15, self).__init__()

        self.conv0h = nn.Conv2d(in_channel, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2),
                                groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 6), groups=dim,
                                        dilation=2)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(6, 0), groups=dim,
                                        dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0h(x)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv0v(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv_spatial_h(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv_spatial_v(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        attn = self.conv1(attn)
        attn = self.relu(attn)
        attn = self.bn(attn)

        y = u * attn
        y = self.relu(y)
        y = self.bn(y)

        return y


#11*11
class ElevenConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ElevenConv2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.block(x)


#15*15
class WideScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(WideScopeConv2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.block(x)


#15*15
class FifteenConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(FifteenConv2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.block(x)


#17*17
class SeventeenConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(SeventeenConv2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=5, dilation=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.block(x)


#5*5，9*9,13*13
class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters, dilation_rate=1):
        super(ResNetConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.batchnorm1 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.batchnorm2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.final_bn = nn.BatchNorm2d(filters)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        # print("asdxgffa", x1.shape)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        # x = F.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        # print("afafa", x.shape)
        # x1 = F.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=False)  # 调整大小
        x_final = x + x1
        x_final = self.final_bn(x_final)
        return F.relu(x_final)


#RB改,激活函数relu改为mish,改变rb与mish位置
# class ResNetConv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, dilation_rate=1):
#         super(ResNetConv2DBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=dilation_rate)
#         self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
#         self.batchnorm1 = nn.BatchNorm2d(filters)
#         self.mish = nn.Mish()
#         self.relu = nn.ReLU(inplace=True)
#         self.final_bn = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x1 = self.batchnorm1(x1)
#         x1 = self.mish(x1)
#
#         x2 = self.conv2(x)
#         x2 = self.batchnorm1(x2)
#         x2 = self.mish(x2)
#
#         x2 = self.conv3(x2)
#         x2 = self.batchnorm1(x2)
#         x2 = self.mish(x2)
#
#         x_final = x1 + x2
#         x_final = self.final_bn(x_final)
#         x_final = self.mish(x_final)
#         return x_final

#RB改RepVGG,激活函数relu改为mish
# class ResNetConv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, dilation_rate=1):
#         super(ResNetConv2DBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=dilation_rate)
#         self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate)
#         self.batchnorm = nn.BatchNorm2d(filters)
#         self.mish = nn.Mish()
#         self.final_bn = nn.BatchNorm2d(filters)
#
#     def forward(self, x):
#         x1 = self.batchnorm(x)
#
#         x2 = self.conv1(x)
#         x2 = self.batchnorm(x2)
#
#         x3 = self.conv2(x)
#         x3 = self.batchnorm(x3)
#         x3 = self.conv3(x3)
#         x3 = self.batchnorm(x3)
#
#         x_final = x1 + x2 + x3
#         x_final = self.mish(x_final)
#         x_final = self.final_bn(x_final)
#         return x_final


#残差块
# class ResNetConv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters,dilation_rate=1):
#         super().__init__()
#
#         self.in_layers = nn.Sequential(
#             nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=dilation_rate),
#             nn.Mish(),
#             nn.GroupNorm(34, filters)
#         )
#
#         self.out_layers = nn.Sequential(
#             nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate),
#             nn.Mish(),
#             nn.GroupNorm(34, filters)
#         )
#
#         if filters == in_channels:
#             self.skip = nn.Identity()
#         else:
#             self.skip = nn.Conv2d(in_channels, filters, kernel_size=1, dilation=dilation_rate)
#
#     def forward(self, x):
#         h = self.in_layers(x)
#         h = self.out_layers(h)
#         return h + self.skip(x)

#11*11

class DoubleConvolutionWithBatchNormalization(nn.Module):
    def __init__(self, in_channels, filters, dilation_rate=1):
        super(DoubleConvolutionWithBatchNormalization, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=dilation_rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=dilation_rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.block(x)
