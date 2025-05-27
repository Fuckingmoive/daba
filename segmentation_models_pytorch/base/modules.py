import torch.nn as nn
import torch.nn.functional as F

try:
    from inplace_abn.abn import InPlaceABN
except ImportError:
    InPlaceABN = None
#from inplace_abn.abn import InPlaceABN


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        #use_batchnorm="inplace"
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

# SCSE注意力模块
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


#############################################################################################################
#CBAM

import torch.nn as nn
import torch

from mmengine.model import BaseModule

__all__ = ['CBAM']


class CBAM(BaseModule):
    """
    The combination of channel attention and spatial attention,
    using average pooling and maximum pooling to aggregate channels and spaces along different dimensions
    """

    def __init__(
            self,
            in_chans: int,
            reduction: int = 16,
            kernel_size: int = 7,
            min_channels: int = 8,
    ):
        super(CBAM, self).__init__()
        # channel-wise attention
        hidden_chans = max(in_chans // reduction, min_channels)
        self.mlp_chans = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, in_chans, kernel_size=1, bias=False),
        )
        # space-wise attention
        self.mlp_spaces = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1)
        avg_x_s = x.mean((2, 3), keepdim=True)
        max_x_s = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = x * self.gate(self.mlp_chans(avg_x_s) + self.mlp_chans(max_x_s))

        # (B, 1, H, W)
        avg_x_c = x.mean(dim=1, keepdim=True)
        max_x_c = x.max(dim=1, keepdim=True)[0]
        x = x * self.gate(self.mlp_spaces(torch.cat((avg_x_c, max_x_c), dim=1)))
        return x


#############################################################################################################
class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, attention_type, **params):
        super().__init__()

        if attention_type is None:
            self.attention = nn.Identity(**params)
        elif attention_type == "scse":
            self.attention = SCSEModule(**params)
        elif attention_type == "cbam":
            self.attention = CBAM(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(attention_type))

    def forward(self, x):
        x = self.attention(x)
        return x
