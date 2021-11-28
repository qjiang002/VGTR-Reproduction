import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention2d(nn.Module):
    def __init__(self, lang_dim, reduction_ratio, K, temperature, init_weight=True):
        """
        Args:
            lang_dim: dimension of sentence embedding
            reduction_ratio: reduction ratio of the bottleneck in fc1 and fc2.
            For more details see https://arxiv.org/pdf/1709.01507.pdf
            K: dimension of the output attention weights
            temperature:
            init_weight:
        """
        super(Attention2d, self).__init__()
        assert temperature % 3 == 1  # Only because paper does 30 -> 1 annealing
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        if lang_dim != 3:
            hidden_planes = lang_dim // reduction_ratio
        else:
            hidden_planes = K
        # self.fc1 = nn.Conv2d(in_planes, hidden_planes, (1, 1), bias=False) # Same as nn.Linear, but probably slower?
        self.fc1 = nn.Linear(lang_dim, hidden_planes, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        # self.fc2 = nn.Conv2d(hidden_planes, K, (1, 1), bias=True) # Unsure why bias=True
        self.fc2 = nn.Linear(hidden_planes, K, bias=True)

        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        """
        Temperature annealing, i.e. reducing τ from 30 to 1 linearly in the first 10 epochs,
        can further improve the top-1 accuracy
        """
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, lang):
        """
        Args:
            lang: N, H  (H is the size of sentence representation vector)

        Returns:
        """
        # x = self.avgpool(x) # N, C, 1, 1
        # x = x.view(x.size()[0], -1) # N, C

        x = self.fc1(lang)  # N, H/r
        x = F.relu(x)
        x = self.fc2(x)  # N, K
        return F.softmax(x / self.temperature, 1)


class DynamicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 groups=1, lang_dim=768, reduction_ratio=4, K=4, temperature=31, init_weight=True):
        super(DynamicConv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.lang_dim = lang_dim
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K

        self.attention = Attention2d(
            lang_dim, reduction_ratio, K, temperature)
        self.candidate_kernel_weight = \
            nn.Parameter(torch.randn(K, out_planes, in_planes //
                         groups, kernel_size, kernel_size), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.candidate_kernel_weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, lang):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        batch_size, in_planes, height, width = x.size()

        softmax_attention = self.attention(lang)  # (N, K)

        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积 (1, N * C, H, W)

        # (K, out_planes * C // groups * kernel_size * kernel_size)
        candidate_kernel_weight = self.candidate_kernel_weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregated_weight = torch.mm(
            softmax_attention, candidate_kernel_weight)

        # (N * out_planes // groups, C,  kernel_size, kernel_size)
        aggregated_weight = aggregated_weight.view(
            -1, self.in_planes, self.kernel_size, self.kernel_size)

        aggregated_bias = torch.mm(
            softmax_attention, self.bias).view(-1) if self.bias is not None else None

        # By default, self.groups is 1
        # The "groups" parameter in conv2d divides the input channel into 1 * N groups (N is the batch size).
        # So each example in a batch is convolved with its own set of aggregated filters.
        # output shape: (1, N * out_planes, H, W)
        output = F.conv2d(x, weight=aggregated_weight, bias=aggregated_bias, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        _, _, output_height, output_width = output.size()
        output = output.view(batch_size, self.out_planes,
                             output_height, output_width)

        # TODO BN, activation
        return output


if __name__ == '__main__':
    batch_size, in_planes, height, width = 32, 64, 20, 20
    x = torch.randn(batch_size, in_planes, height, width)

    out_planes, kernel_size = 16, 3
    model = DynamicConv2d(in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, bias=True,
                          groups=1, reduction_ratio=4, K=4, temperature=31, init_weight=True)
    print(x.shape)
    print(model(x).shape)
    model.update_temperature()
    print(model(x).shape)
