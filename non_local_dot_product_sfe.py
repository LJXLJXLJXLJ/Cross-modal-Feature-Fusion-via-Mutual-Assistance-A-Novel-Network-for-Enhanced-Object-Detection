import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels=1024, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

import torch
from torch import nn
from torch.nn import functional as F


class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x, cond):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return x * (scale + 1) + shift


class SpatialFeatureEnhance(nn.Module):

    def __init__(self, window_size=7, in_channels=3, out_channels=3, inter_channels=6):
        super(SpatialFeatureEnhance, self).__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft = SFTLayer(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, query_data, key_data):
        b, c, h, w = key_data.shape
        x = key_data.clone()
        query_data = query_data.permute(0, 2, 3, 1).reshape(-1, c, 1)
        key_data = F.unfold(key_data, kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
        key_data = key_data.reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(query_data.permute(0, 2, 1), key_data) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        out = self.sft(x, sim_map)
        return out

class NonLocalBlockND_SFE(nn.Module):
    def __init__(self, in_channels=1024, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND_SFE, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.sfe=SpatialFeatureEnhance(window_size=7, in_channels=1024, out_channels=1024, inter_channels=6)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # z = W_y + x
        z=self.sfe(W_y,x)
        return z

# class NONLocalBlock1D(_NonLocalBlockND):
#     def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
#         super(NONLocalBlock1D, self).__init__(in_channels,
#                                               inter_channels=inter_channels,
#                                               dimension=1, sub_sample=sub_sample,
#                                               bn_layer=bn_layer)
#
#
# class NONLocalBlock2D(_NonLocalBlockND):
#     def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
#         super(NONLocalBlock2D, self).__init__(in_channels,
#                                               inter_channels=inter_channels,
#                                               dimension=2, sub_sample=sub_sample,
#                                               bn_layer=bn_layer)
#
#
# class NONLocalBlock3D(_NonLocalBlockND):
#     def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
#         super(NONLocalBlock3D, self).__init__(in_channels,
#                                               inter_channels=inter_channels,
#                                               dimension=3, sub_sample=sub_sample,
#                                               bn_layer=bn_layer)


if __name__ == '__main__':


    # for (sub_sample, bn_layer) in [(True, True)]:
        # img = torch.zeros(2, 3, 20)
        # net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        # out = net(img)
        # print(out.size())

    img = torch.zeros(2, 1024, 20, 20)
    net = NonLocalBlockND_SFE(1024)
    out = net(img)
    print(out.size())

        # img = torch.randn(2, 3, 8, 20, 20)
        # net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        # out = net(img)
        # print(out.size())



