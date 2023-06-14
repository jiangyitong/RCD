import torch
import torch.nn as nn
import torch.nn.functional as F


class BD(nn.Module):
    def __init__(self, G=10):
        super(BD, self).__init__()

        self.momentum = 0.9
        self.register_buffer('running_wm', torch.eye(G).expand(G, G))

    def forward(self, x, T=5, eps=1e-5):
        N, G, C, H, W = x.size()
        x_in = x.transpose(0, 1).contiguous().view(G, -1)
        if self.training:
            mean = x_in.mean(-1, keepdim=True)
            xc = x_in - mean
            d, m = x_in.size()
            P = [None] * (T + 1)
            P[0] = torch.eye(G, device=x.device)
            Sigma = (torch.matmul(xc, xc.transpose(0, 1))) / float(m) + P[0] * eps
            rTr = (Sigma * P[0]).sum([0, 1], keepdim=True).reciprocal()
            Sigma_N = Sigma * rTr
            wm = torch.linalg.solve_triangular(
                torch.linalg.cholesky(Sigma_N), P[0], upper=False
            )
            self.running_wm = self.momentum * self.running_wm + (1 - self.momentum) * wm
        else:
            wm = self.running_wm

        x_out = wm @ x_in
        x_out = x_out.view(G, N, C, H, W).transpose(0, 1).contiguous()

        return x_out


def rb_decorrelatio(x, T=8, eps=1e-8):
    N, G, C, H, W = x.size()
    x_in = x.reshape(N, G, -1)
    mean = x_in.mean(-1, keepdim=True)
    xc = x_in - mean
    n, d, m = x_in.size()
    P = [None] * (T + 1)
    P[0] = torch.eye(G, device=x.device).expand(N, G, G)
    Sigma = (xc @ xc.transpose(1, 2)) / float(m) + P[0] * eps
    rTr = (Sigma * P[0]).sum([1, 2], keepdim=True).reciprocal()
    Sigma_N = Sigma * rTr
    for k in range(T):
        mat_power3 = P[k] @ P[k] @ P[k]
        P[k + 1] = 1.5 * P[k] - 0.5 * (mat_power3 @ Sigma_N)

    wm = P[T]
    x_out = wm @ x_in
    x_out = x_out.view(N, G, C, H, W)
    return x_out


def cov(x, eps=1e-5):
    N, G, C, H, W = x.size()
    x_in = x.transpose(0, 1).contiguous().view(G, -1)
    mean = x_in.mean(-1, keepdim=True)
    xc = x_in - mean
    d, m = x_in.size()
    Sigma = (xc @ xc.transpose(0, 1)) / float(m) + torch.eye(G, device=x.device) * eps
    return Sigma


def ncov(x, eps=1e-5):
    N, G, C, H, W = x.size()
    x_in = x.contiguous().view(N, G, -1)
    mean = x_in.mean(-1, keepdim=True)
    xc = x_in - mean
    n, d, m = x_in.size()
    Sigma = (torch.matmul(xc, xc.transpose(1, 2))) / float(m) + torch.eye(G, device=x.device).expand(n, -1, -1) * eps
    return Sigma


def cov_sum(x):
    x = x[0].unsqueeze(0)
    N, G, _, _, _ = x.shape
    x_cov = cov(x)
    M = torch.ones(N, G, G).to(x.device) - torch.eye(G, device=x.device).expand(N, G, G)
    diag_mean = (torch.eye(G, device=x.device).expand(N, G, G) * x_cov).mean()
    return ((x_cov * M) ** 2).sum() / diag_mean


def ncov_sum(x):
    x = x[0].unsqueeze(0)
    N, G, _, _, _ = x.shape
    x_cov = ncov(x)
    M = torch.ones(N, G, G).to(x.device) - torch.eye(G, device=x.device).expand(N, G, G)
    diag_mean = (torch.eye(G, device=x.device).expand(N, G, G) * x_cov).mean([-2, -1])
    return (((x_cov * M) ** 2).sum([-2, -1]) / diag_mean).mean()


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, mf2f=False):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=mf2f),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=mf2f),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, mf2f=False):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames * (3 + 1), num_in_frames * self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames, bias=mf2f),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=mf2f),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch, mf2f=False):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=mf2f),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch, mf2f=mf2f)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch, mf2f=False):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch, mf2f=mf2f),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=mf2f),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, mf2f=False, group=1):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * group, kernel_size=3, padding=1, bias=mf2f),
            nn.BatchNorm2d(in_ch * group),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * group, out_ch, kernel_size=3, padding=1, bias=mf2f, groups=group)
        )

    def forward(self, x):
        return self.convblock(x)


class RBBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=3, mf2f=False, out=3, ratio=1, group=12, rep=False):

        super(RBBlock, self).__init__()
        self.chs_lyr0 = 32 * ratio
        self.chs_lyr1 = 64 * ratio
        self.chs_lyr2 = 128 * ratio
        self.rep = rep
        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0, mf2f=mf2f)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, mf2f=mf2f)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, mf2f=mf2f)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, mf2f=mf2f)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, mf2f=mf2f)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out, mf2f=mf2f, group=group)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, noise_map):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(torch.cat((in0, noise_map), dim=1))
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        # Estimation
        x = self.outc(x0 + x1)

        if self.rep:
            x = in0[:, :3] - x

        return x


class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=3, mf2f=False, out=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0, mf2f=mf2f)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, mf2f=mf2f)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, mf2f=mf2f)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, mf2f=mf2f)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, mf2f=mf2f)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out, mf2f=mf2f)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        # Estimation
        x = self.outc(x0 + x1)

        # Residual
        x = in1 - x

        return x


class Refine(nn.Module):
    def __init__(self, in_ch):
        super(Refine, self).__init__()
        self.net1 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


def cov_img(a, b):
    N, C, H, W = a.shape
    fuse = torch.cat([a.view(N, 1, C, H, W), b.view(N, 1, C, H, W)], dim=1)
    return cov_sum(fuse)
