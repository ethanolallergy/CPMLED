import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.submodules import *
from basicsr.utils.registry import ARCH_REGISTRY

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool

class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g  # (w + g) #self.gate(x, w)

class PHA(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.sa=PAM_Module(in_ch)
        self.ca=LocalAttention(in_ch)
        # self.pa=PixelAttention(in_ch)
    def forward(self,x):
        x=self.sa(x)+self.ca(x)+x
        return x

class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()


    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class CCA(nn.Module):
    def __init__(self,channel,b=1, gamma=2, n_curve=3):
        super(CCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        input = self.relu(input) - self.relu(input - 1)
        for i in range(self.n_curve):
            input = input + out[:, i:i+1] * input * (1 - input)

        return input


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))#[bs,reduction_dim,bin,bin]
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):#[bs,in_dim,h,w]
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))#[bs,reduction_dim,h,w]
        out_feat = self.fuse(torch.cat(out, 1))#[bs,in_dim+reduction_dim*4,h,w]->[bs,in_dim,h,w]
        return out_feat#[bs,in_dim,h,w]


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                Downsample(channels=in_channels, filt_size=3,stride=2),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):#[bs,c,h,w]
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out#[bs,2c,0.5h,0.5w]

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):#[bs,c,h,w]
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out#[bs,2c,2h,2w]


class BasicBlock_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_E, self).__init__()
        self.mode = mode

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU()
        )
        if mode == 'down':
            self.reshape_conv = ResidualDownSample(out_channels, out_channels)

    def forward(self, x):#[bs,c,h,w]
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        if self.mode is not None:
            out = self.reshape_conv(out)
        return out#[bs,2c,0.5h,0.5w]

class MSFE(nn.Module):
    def __init__(self, inchannel):
        super(MSFE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.convmix = nn.Sequential(nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)

        return out


class BasicBlock_E_MSFE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_E_MSFE, self).__init__()
        self.mode = mode

        self.body1 = nn.Sequential(
            MSFE(inchannel=in_channels)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU()
        )
        if mode == 'down':
            self.reshape_conv = ResidualDownSample(out_channels, out_channels)

    def forward(self, x):#[bs,c,h,w]
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        if self.mode is not None:
            out = self.reshape_conv(out)
        return out#[bs,2c,0.5h,0.5w]


class BasicBlock_D_2ResMSFE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_D_2ResMSFE, self).__init__()
        self.mode = mode
        if mode == 'up':
            self.reshape_conv = ResidualUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            MSFE(inchannel=out_channels)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias)
        )

    def forward(self, x):#[bs,c,h,w]
        if self.mode is not None:
            x = self.reshape_conv(x)
        res1 = self.body1(x)
        out1 = res1 + x
        res2 = self.body2(out1)
        out2 = res2 + out1
        return out2#[bs,2c,2h,2w]

@ARCH_REGISTRY.register()
class CPMLED(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], connection=False,
                 # LQ_stage=True,scale_factor=1,
                 # use_semantic_loss=True,codebook_params=[128],
                 # gt_resolution=256
                 ):
        super(CPMLED, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        self.connection = connection
        self.E_block0 = BasicBlock_E(3, ch1)
        self.E_block1 = nn.Sequential(
            nn.PReLU(),
            BasicBlock_E(ch1, ch1),
            BasicBlock_E(ch1, ch2, mode='down')
            )
        self.E_block2 = nn.Sequential(BasicBlock_E(ch2, ch2),
                                      BasicBlock_E(ch2, ch3, mode='down'))
        self.E_block3 = nn.Sequential(BasicBlock_E(ch3, ch3),
                                      BasicBlock_E(ch3, ch4, mode='down'))
        
        self.side_out = nn.Conv2d(ch4, 3, 3, stride=1, padding=1)

        self.M_block = PHA(ch4)

        # dynamic filter
        ks_2d = 5
        self.conv_fac_k3 = nn.Sequential(
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4 * ks_2d ** 2, 1, stride=1))

        self.conv_fac_k2 = nn.Sequential(
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3 * ks_2d ** 2, 1, stride=1))

        self.conv_fac_k1 = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2 * ks_2d ** 2, 1, stride=1))

        self.kconv_deblur = KernelConv2D(ksize=ks_2d, act=True)


        # curve
        self.curve_n = 3
        self.conv_1c = CCA(ch2, self.curve_n)
        self.conv_2c = CCA(ch3, self.curve_n)
        self.conv_3c = CCA(ch4, self.curve_n)
        self.PPM1 = PPM(ch2, ch2 // 4, bins=(1, 2, 3, 6))
        self.PPM2 = PPM(ch3, ch3 // 4, bins=(1, 2, 3, 6))
        self.PPM3 = PPM(ch4, ch4 // 4, bins=(1, 2, 3, 6))
        # Decoder D
        # self.D_block3 = BasicBlock_D_2Res(ch4, ch4)
        self.D_block3 = nn.Sequential(BasicBlock_D_2ResMSFE(ch4, ch4),
                                      BasicBlock_E_MSFE(ch4, ch4))

        self.D_block2 = nn.Sequential(BasicBlock_D_2ResMSFE(ch4, ch3, mode='up'),
                                      BasicBlock_E_MSFE(ch3, ch3))
        # self.D_block1 = BasicBlock_D_2Res(ch3, ch2, mode='up')
        self.D_block1 = nn.Sequential(BasicBlock_D_2ResMSFE(ch3, ch2, mode='up'),
                                      BasicBlock_E_MSFE(ch2, ch2))
        self.D_block0 = nn.Sequential(
            BasicBlock_D_2ResMSFE(ch2, ch1, mode='up'),
            BasicBlock_E_MSFE(ch1, ch1)
            )
        # self.D_block=nn.Conv2d(ch1, 3, 3, stride=1, padding=1)
        self.D_block = BasicBlock_E_MSFE(ch1, 3)


    def forward(self, x, side_loss=True):
        # Encoder
        # shallow_feature=self.shallow_feature_extraction(x)
        e_feat0 = self.E_block0(x)
        # e_conv = self.Econv(e_feat0)
        e_feat1 = self.E_block1(e_feat0)  # 64 1/2
        e_feat1 = self.PPM1(e_feat1)
        e_feat1 = self.conv_1c(e_feat1)

        e_feat2 = self.E_block2(e_feat1)  # 128 1/4
        e_feat2 = self.PPM2(e_feat2)
        e_feat2 = self.conv_2c(e_feat2)

        e_feat3 = self.E_block3(e_feat2)  # 256 1/8
        e_feat3 = self.PPM3(e_feat3)
        e_feat3 = self.conv_3c(e_feat3)

        if side_loss:
            out_side = self.side_out(e_feat3)

        # Mid
        m_feat = self.M_block(e_feat3)

        # Decoder
        d_feat3 = self.D_block3(m_feat)  # 256 1/8
        kernel_3 = self.conv_fac_k3(e_feat3)
        d_feat3 = self.kconv_deblur(d_feat3, kernel_3)
        # d_feat3 = self.fusion3(d_feat3, e_feat3)
        if self.connection:
            d_feat3 = d_feat3 + e_feat3


        d_feat2 = self.D_block2(d_feat3)  # 128 1/4
        kernel_2 = self.conv_fac_k2(e_feat2)
        d_feat2 = self.kconv_deblur(d_feat2, kernel_2)
        # d_feat2 = self.fusion2(d_feat2, e_feat2)
        if self.connection:
            d_feat2 = d_feat2 + e_feat2


        d_feat1 = self.D_block1(d_feat2)  # 64 1/2
        kernel_1 = self.conv_fac_k1(e_feat1)
        d_feat1 = self.kconv_deblur(d_feat1, kernel_1)
        if self.connection:
            d_feat1 = d_feat1 + e_feat1

        # d_feat1 = self.fusion1(d_feat1, e_feat1)
        out = self.D_block0(d_feat1)
        out = self.D_block(out)
        out = out + x

        if side_loss:
            # return out_side, out
            return out, out_side
        else:
            return out
