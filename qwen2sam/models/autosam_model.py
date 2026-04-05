"""
AutoSAM model components adapted from https://github.com/talshaharabany/AutoSAM

Self-contained: HarDNet backbone + SmallDecoder + ModelEmb.
Produces dense prompt embeddings (B, 256, 64, 64) for SAM's mask decoder,
replacing the manual prompt encoder with learned image embeddings.

Reference: "AutoSAM: Adapting SAM to Medical Images by Overloading the
Prompt Encoder" (Shaharabany et al., 2023) — https://arxiv.org/abs/2306.06370
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================== #
#  Base building blocks (from AutoSAM/models/base.py)                     #
# ===================================================================== #

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.data.size(0), -1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1,
                 dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels,
                                              stride=stride))


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        groups = in_channels
        self.add_module('dwconv', nn.Conv2d(
            groups, groups, kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
                 dropout=0.1, bias=False):
        super().__init__()
        groups = 1
        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride,
            padding=kernel // 2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(True))


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super().__init__()
        P = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in, inx=-1):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out


class UpBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, func=None,
                 drop=0):
        super().__init__()
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in, x_up):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), 1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh':
            return torch.tanh(self.BN(x1))
        elif self.func == 'relu':
            return F.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid':
            return torch.sigmoid(self.BN(x1))
        else:
            return x1


# ===================================================================== #
#  HarDNet backbone (from AutoSAM/models/hardnet.py)                      #
# ===================================================================== #

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers,
                 keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(
                i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDNet(nn.Module):
    def __init__(self, depth_wise=False, arch=85, pretrained=True,
                 weight_path=''):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(ConvLayer(
            in_channels=3, out_channels=first_ch[0], kernel=3, stride=2,
            bias=False))
        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1],
                                   kernel=second_kernel))
        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)
            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.base.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(ch, 1000)))

        # Set architecture-specific attributes (needed by decoder)
        if arch == 39:
            self.features = 640
            self.full_features = [48, 96, 320, 640, 1024]
            self.list = [1, 4, 7, 10, 13]
            trim_len = 11
        elif arch == 68:
            self.features = 1024
            self.full_features = [64, 128, 320, 640, 1024]
            self.list = [1, 4, 9, 12, 15]
            trim_len = 16
        else:  # arch == 85
            self.features = 1280
            self.full_features = [96, 192, 320, 720, 1280]
            self.list = [1, 4, 9, 14, 18]
            trim_len = 19

        if pretrained:
            if hasattr(torch, 'hub'):
                if arch == 68 and not depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth'
                elif arch == 85 and not depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth'
                elif arch == 68 and depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth'
                else:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth'

                self.load_state_dict(torch.hub.load_state_dict_from_url(
                    checkpoint, progress=False,
                    map_location=torch.device('cpu')))
            else:
                postfix = 'ds' if depth_wise else ''
                weight_file = '%shardnet%d%s.pth' % (weight_path, arch,
                                                      postfix)
                if not os.path.isfile(weight_file):
                    raise FileNotFoundError(f'{weight_file} not found')
                weights = torch.load(weight_file, map_location='cpu')
                self.load_state_dict(weights)

            postfix = 'DS' if depth_wise else ''
            print(f'ImageNet pretrained HarDNet{arch}{postfix} loaded')

        # Trim classifier layers (keep only feature extraction)
        self.base = self.base[0:trim_len]

    def forward(self, x):
        for inx, layer in enumerate(self.base):
            x = layer(x)
            if inx == self.list[0]:
                x2 = x
                if inx == len(self.base) - 1:
                    return x2
            elif inx == self.list[1]:
                x4 = x
                if inx == len(self.base) - 1:
                    return x2, x4
            elif inx == self.list[2]:
                x8 = x
                if inx == len(self.base) - 1:
                    return x2, x4, x8
            elif inx == self.list[3]:
                x16 = x
                if inx == len(self.base) - 1:
                    return x2, x4, x8, x16
            elif inx == self.list[4]:
                x32 = x
                if inx == len(self.base) - 1:
                    return x2, x4, x8, x16, x32


# ===================================================================== #
#  SmallDecoder + ModelEmb (from AutoSAM/models/model_single.py)          #
# ===================================================================== #

class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super().__init__()
        self.up1 = UpBlockSkip(
            full_features[3] + full_features[2], full_features[2],
            func='relu', drop=0)
        self.up2 = UpBlockSkip(
            full_features[2] + full_features[1], full_features[1],
            func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = torch.tanh(self.final(z))
        return out


class ModelEmb(nn.Module):
    """
    Learned dense prompt embedding generator for SAM.

    Input: (B, 3, Idim, Idim) image
    Output: (B, 256, 64, 64) dense embeddings for SAM's mask decoder

    Uses HarDNet backbone (pretrained on ImageNet) + SmallDecoder.
    """

    def __init__(self, arch=85, depth_wise=False, pretrained=True):
        super().__init__()
        self.backbone = HarDNet(depth_wise=depth_wise, arch=arch,
                                pretrained=pretrained)
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=256)

    def forward(self, img):
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(
            dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings


# ===================================================================== #
#  AutoSAM inference helpers (adapted from AutoSAM/inference.py)          #
# ===================================================================== #

def norm_batch(x):
    """Normalize mask logits to [0, 1] per-sample."""
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_val = x.view(bs, -1).min(dim=1)[0].view(bs, 1, 1, 1)
    max_val = x.view(bs, -1).max(dim=1)[0].view(bs, 1, 1, 1)
    return (x - min_val) / (max_val - min_val + 1e-6)


def sam_call(batched_input, sam, dense_embeddings):
    """
    Run SAM with learned dense embeddings (no manual prompts).

    Args:
        batched_input: list of dicts with 'image' key (preprocessed SAM images)
        sam: SAM model (frozen)
        dense_embeddings: (B, 256, 64, 64) from ModelEmb

    Returns:
        low_res_masks: (B, 1, 256, 256) raw mask logits
    """
    input_images = torch.stack(
        [sam.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sparse_embeddings_none, _ = sam.prompt_encoder(
        points=None, boxes=None, masks=None)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def dice_loss(y_pred, y_true, smooth=1e-6):
    """Dice loss (Tversky with alpha=beta=0.5)."""
    y_pred = y_pred.clamp(0, 1)
    y_true = y_true.clamp(0, 1)
    dims = tuple(range(1, y_pred.ndim))
    tp = (y_true * y_pred).sum(dim=dims)
    fn = (y_true * (1 - y_pred)).sum(dim=dims)
    fp = ((1 - y_true) * y_pred).sum(dim=dims)
    tversky = (tp + smooth) / (tp + 0.5 * fn + 0.5 * fp + smooth)
    return 1 - tversky.mean()
