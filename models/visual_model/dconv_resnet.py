import torchvision.models as models
import torch
import torch.nn.functional as F
from .dconv import DynamicConv2d
from torch import nn

from models.visual_model import dconv

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, lang_dim, K, stride=1, groups=1, dilation=1):
    return DynamicConv2d(in_planes, out_planes, lang_dim=lang_dim, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, K=K)

def conv7x7(in_planes, out_planes, lang_dim, K, stride=1, groups=1, dilation=1):
    return DynamicConv2d(in_planes, out_planes, lang_dim=lang_dim, kernel_size=7, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, K=K)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1(in_planes, out_planes, lang_dim, K, stride=1):
    return DynamicConv2d(in_planes, out_planes, lang_dim=lang_dim, kernel_size=1, stride=stride, bias=False, K=K)

class DConvConv2d(nn.Conv2d):
    def forward(self, x, lang=None):
        return super(DConvConv2d, self).forward(x)

class DConvBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x, lang=None):
        return super(DConvBatchNorm2d, self).forward(x)

class DConvReLU(nn.ReLU):
    def forward(self, x, lang=None):
        return super(DConvReLU, self).forward(x)

class DConvMaxPool2d(nn.MaxPool2d):
    def forward(self, x, lang=None):
        return super(DConvMaxPool2d, self).forward(x)

class DConvAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, x, lang=None):
        return super(DConvAdaptiveAvgPool2d, self).forward(x)

class DConvLinear(nn.Linear):
    def forward(self, x, lang=None):
        return super(DConvLinear , self).forward(x)

class DConvSequential(nn.Sequential):
    def forward(self, x, lang=None):
        for module in self._modules.values():
            x = module(x, lang)
        return x



class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, lang_dim, K, dconv_places, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = DConvBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if dconv_places in ['first_in_block', 'both']:
            self.conv1 = conv3x3(inplanes, planes, lang_dim, K, stride=stride)
        else:
            self.conv1 = DConvConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=3, bias=False)

        self.bn1 = norm_layer(planes)
        self.relu = DConvReLU(inplace=True)

        if dconv_places in ['last_in_block', 'both']:
            self.conv2 = conv3x3(inplanes, planes, lang_dim, K, stride=stride)
        else:
            self.conv2 = DConvConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=3, bias=False)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, lang):
        identity = x

        out = self.conv1(x, lang)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, lang)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, lang)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, lang_dim, K, dconv_places, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = DConvBatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1


        if dconv_places in ['first_in_block', 'both']:
            self.conv1 = conv1x1(inplanes, width, lang_dim, K)
        else:
            self.conv1 = DConvConv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, lang_dim, K,
                             stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)

        if dconv_places in ['last_in_block', 'both']:
            self.conv3 = conv1x1(width, planes * self.expansion, lang_dim, K)
        else:
            self.conv3 = DConvConv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = DConvReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, lang):
        identity = x

        out = self.conv1(x, lang)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, lang)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, lang)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x, lang)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, lang_dim, K, dconv_places, dconv_7x7, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = DConvBatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dconv_places = dconv_places
        self.dconv_7x7 = dconv_7x7
        self.dilation = 1
        self.K = K
        self.lang_dim = lang_dim
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if self.dconv_7x7:
            self.conv1 = conv7x7(3, self.inplanes, lang_dim, K, stride=2, dilation=3) # Padding is replaced with dilation
        else:
            self.conv1 = DConvConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = DConvReLU(inplace=True)
        self.maxpool = DConvMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], lang_dim, K, dconv_places)
        self.layer2 = self._make_layer(block, 128, layers[1], lang_dim, K, dconv_places, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], lang_dim, K, dconv_places, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], lang_dim, K, dconv_places, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = DConvAdaptiveAvgPool2d((1, 1))
        self.fc = DConvLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, DConvConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (DConvBatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, DynamicConv2d):
                m.update_temperature()

    def _make_layer(self, block, planes, blocks, lang_dim, K, dconv_places, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DConvSequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        lang_dim, K, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, lang_dim, K, dconv_places, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, lang_dim, K, dconv_places, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return DConvSequential(*layers)

    def _forward_impl(self, x, lang):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, lang)
        x = self.layer2(x, lang)
        x = self.layer3(x, lang)
        x = self.layer4(x, lang)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x, lang):
        return self._forward_impl(x, lang)


def _resnet(arch, block, layers, pretrained, progress, lang_dim, K, dconv_places, dconv_7x7, **kwargs):
    model = ResNet(block, layers, lang_dim, K, dconv_places, dconv_7x7, **kwargs)
    if pretrained:
        raise NotImplemented("Pre-trained dynamic convolution not supported!")
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def resnet18(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def resnet34(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   lang_dim, K, **kwargs)


def resnet50(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def resnet101(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def resnet152(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def resnext50_32x4d(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def resnext101_32x8d(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def wide_resnet50_2(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, lang_dim, K, dconv_places, dconv_7x7, **kwargs)


def wide_resnet101_2(lang_dim, K, dconv_places, dconv_7x7, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, lang_dim, K, dconv_places, dconv_7x7, **kwargs)
