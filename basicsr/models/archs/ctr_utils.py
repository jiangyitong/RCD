
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
    Sigma = (torch.matmul(xc, xc.transpose(1, 2))) / float(m) + torch.eye(G, device=x.device).expand(n,-1,-1) * eps
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
    diag_mean = (torch.eye(G, device=x.device).expand(N, G, G) * x_cov).mean([-2,-1])
    return (((x_cov * M) ** 2).sum([-2,-1]) / diag_mean).mean()

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





class RBDVDnet(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, mf2f=False):
        super(RBDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp = RBBlock(num_input_frames=1, mf2f=mf2f, out=30)
        self.map = nn.Conv2d(30, 10, kernel_size=3, stride=1, padding=1)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, noise_map=None, accu_map=None):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''

        # hack
        N, C, H, W = img.shape

        if noise_map is None:
            noise_map = torch.randn(1, 1, H, W)
        if accu_map is None:
            accu_map = torch.randn(1, 10, H, W)

        x = img
        if (H % 4 != 0):
            x = F.pad(x, [0, 0, 4 - (H % 4), 0], mode='reflect')
            noise_map = F.pad(noise_map, [0, 0, 4 - (H % 4), 0], mode='reflect')
        if (W % 4 != 0):
            x = F.pad(x, [4 - (W % 4), 0, 0, 0], mode='reflect')
            noise_map = F.pad(noise_map, [4 - (W % 4), 0, 0, 0], mode='reflect')

        # Unpack inputs

        out, q_loss = self.temp(x, noise_map)
        # unhack
        N1, C1, H1, W1 = out.shape
        if (H % 4 != 0):
            out = out[:, :, (4 - (H % 4)):H1, 0:W1]
            noise_map = noise_map[:, :, (4 - (H % 4)):H1, 0:W1]
        if (W % 4 != 0):
            out = out[:, :, 0:H, (4 - (W % 4)):W1]
            noise_map = noise_map[:, :, 0:H, (4 - (W % 4)):W1]

        noise = out

        pre_map = self.map(out)

        curr_map = pre_map * (accu_map * 10).softmax(1) * 10
        codebook = (torch.arange(10).float().to(x.device) + 1) ** 2
        map = curr_map.clamp(max=1e5).softmax(1) * (codebook.view(1, -1, 1, 1))
        # map  = (map.view(-1, 10, 1, H, W) * noise.view(N, 10, C, H, W)).sum(1)
        noise = noise.view(N, 10, C, H, W) / codebook.mean() * map.view(-1, 10, 1, H, W)
        std_est = noise.detach().view(N, 10, C, H, W) / codebook.mean() * map.view(-1, 10, 1, H, W)
        x = img - noise.sum(1)

        return x, q_loss, curr_map



class RBDVDnet_DCR(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, mf2f=False, level=12):
        super(RBDVDnet_DCR, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp = RBBlock(num_input_frames=1, mf2f=mf2f, out=3*level, ratio=1)
        self.map = nn.Conv2d(3*level+1, level, kernel_size=3, stride=1, padding=1)
        self.bd = BD(G=level)
        self.reset_params()

        self.level = level
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, noise_map=None, accu_map=None):

        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''

        # hack
        N, C, H, W = img.shape

        x = img
        if (H % 4 != 0):
            x = F.pad(x, [0, 0, 4 - (H % 4), 0], mode='reflect')
            noise_map = F.pad(noise_map, [0, 0, 4 - (H % 4), 0], mode='reflect')
        if (W % 4 != 0):
            x = F.pad(x, [4 - (W % 4), 0, 0, 0], mode='reflect')
            noise_map = F.pad(noise_map, [4 - (W % 4), 0, 0, 0], mode='reflect')

        # Unpack inputs
        level = self.level
        out = self.temp(x, torch.zeros_like(noise_map))
        # unhack
        N1, C1, H1, W1 = out.shape
        if (H % 4 != 0):
            out = out[:, :, (4 - (H % 4)):H1, 0:W1]
            noise_map = noise_map[:, :, (4 - (H % 4)):H1, 0:W1]
        if (W % 4 != 0):
            out = out[:, :, 0:H, (4 - (W % 4)):W1]
            noise_map = noise_map[:, :, 0:H, (4 - (W % 4)):W1]
        out = out.view(N, level, C, H, W)

        codebook = (torch.arange(level).float().to(x.device) + 1)

        # out = rb_decorrelatio(out)

        # out = tri_decorrelation_b(out)

        out = self.bd(out)

        out = (out - out.mean([2, 3, 4], keepdims=True)) / (out.std([2, 3, 4], keepdims=True) + 1e-8) * (60 // level ) / 255

        out = out * codebook.view(1, -1, 1, 1, 1)

        noise = out

        # cov_reg = (cov(noise) ** 2).sum() * H * W

        cov_reg = torch.zeros([]).to(x.device)

        pre_map = self.map(torch.cat([out.view(N, -1, H, W), torch.zeros_like(noise_map)], dim=1))

        pre_map_down = nn.AdaptiveAvgPool2d((H // 8, W // 8))(pre_map)

        mean_map = (pre_map_down.mean(1, keepdims=True) * 20).sigmoid() * level

        # mean_map = noise_map * 255 / 10

        linear_x = 1 / ((codebook.view(1, -1, 1, 1) - mean_map) ** 2 + 1e-8)

        pre_map = linear_x.softmax(1)

        pre_map_up = F.interpolate(pre_map, (H, W))

        # if accu_map is None:
        #     accu_map = torch.ones_like(pre_map_up)
        # curr_map = (pre_map_up) * accu_map
        # curr_map = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        map = pre_map_up.mean([2, 3]) * (accu_map * 1).mean([2, 3])
        # map  = curr_map.mean([2, 3]) * (accu_map * 1).mean([2, 3])
        map = map / (map.sum(1, keepdims=True) + 1e-5)
        # map = map.sqrt()
        x_mean = img.view(N, 1, C, H, W) - noise.view(N, level, C, H, W)
        noise_fuse = noise.view(N, level, C, H, W) * map.view(-1, level, 1, 1, 1)
        noise_fuse = noise_fuse.sum(1)
        x = img - noise_fuse

        return x,  x_mean


class RBDVDnet_DCR_All(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, mf2f=False):
        super(RBDVDnet_DCR_All, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp = RBBlock(num_input_frames=1, mf2f=mf2f, out=36)
        self.map = nn.Conv2d(37, 12, kernel_size=3, stride=1, padding=1)
        self.reset_params()
        self.bd = BD(G=12)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, noise_map=None, accu_map=None):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''

        # hack
        N, C, H, W = img.shape

        x = img
        if (H % 4 != 0):
            x = F.pad(x, [0, 0, 4 - (H % 4), 0], mode='reflect')
            noise_map = F.pad(noise_map, [0, 0, 4 - (H % 4), 0], mode='reflect')
        if (W % 4 != 0):
            x = F.pad(x, [4 - (W % 4), 0, 0, 0], mode='reflect')
            noise_map = F.pad(noise_map, [4 - (W % 4), 0, 0, 0], mode='reflect')

        # Unpack inputs

        out = self.temp(x, noise_map)
        # unhack
        N1, C1, H1, W1 = out.shape
        if (H % 4 != 0):
            out = out[:, :, (4 - (H % 4)):H1, 0:W1]
            noise_map = noise_map[:, :, (4 - (H % 4)):H1, 0:W1]
        if (W % 4 != 0):
            out = out[:, :, 0:H, (4 - (W % 4)):W1]
            noise_map = noise_map[:, :, 0:H, (4 - (W % 4)):W1]
        out = out.view(N, 12, C, H, W)

        codebook = (torch.arange(12).float().to(x.device) + 1)

        # out = rb_decorrelatio(out)

        # out = tri_decorrelation_b(out)

        out = self.bd(out)

        out = (out - out.mean([2, 3, 4], keepdims=True)) / (out.std([2, 3, 4], keepdims=True) + 1e-8) * 5 / 255

        out = out * codebook.view(1, -1, 1, 1, 1)

        noise = out

        # cov_reg = (cov(noise) ** 2).sum() * H * W

        cov_reg = torch.zeros([]).to(x.device)

        pre_map = self.map(torch.cat([out.view(N, -1, H, W), noise_map], dim=1))

        pre_map_down = nn.AdaptiveAvgPool2d((H // 2, W // 2))(pre_map)

        mean_map = (pre_map_down.mean(1, keepdims=True) * 20).sigmoid() * 12

        # mean_map = noise_map * 255 / 10

        linear_x = 1 / (((codebook.view(1, -1, 1, 1) - mean_map).abs() + 1e-8) ** 0.5 + 1e-8)

        pre_map = linear_x.softmax(1)

        pre_map_up = F.interpolate(pre_map, (H, W))

        curr_map = (pre_map_up) * accu_map
        curr_map_out = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        # map = pre_map_up.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map  = curr_map.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map = map / (map.sum(1, keepdims=True) + 1e-5)

        # map = map.sqrt()

        map = curr_map_out

        x_mean = img.view(N, 1, C, H, W) - noise.view(N, 12, C, H, W)
        noise_fuse = noise.view(N, 12, C, H, W) * map.view(-1, 12, 1, H, W)

        # noise_corr = ( img.view(N, 1, C, H, W) - noise_fuse).sum(2)
        # curr_map = (curr_map) * noise_corr
        # curr_map = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        noise_fuse = noise_fuse.sum(1)
        x = img - noise_fuse

        return x, curr_map_out.clamp(min=-1e3, max=1e3), x_mean


class RBDVDnet_DCR_2x(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, mf2f=False):
        super(RBDVDnet_DCR_2x, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp = RBBlock(num_input_frames=1, mf2f=mf2f, out=36, ratio=2)
        self.map = nn.Conv2d(37, 12, kernel_size=3, stride=1, padding=1)
        self.bd = BD(G=12)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, noise_map=None, accu_map=None):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''

        # hack
        N, C, H, W = img.shape

        x = img
        if (H % 4 != 0):
            x = F.pad(x, [0, 0, 4 - (H % 4), 0], mode='reflect')
            noise_map = F.pad(noise_map, [0, 0, 4 - (H % 4), 0], mode='reflect')
        if (W % 4 != 0):
            x = F.pad(x, [4 - (W % 4), 0, 0, 0], mode='reflect')
            noise_map = F.pad(noise_map, [4 - (W % 4), 0, 0, 0], mode='reflect')

        # Unpack inputs

        out = self.temp(x, noise_map)
        # unhack
        N1, C1, H1, W1 = out.shape
        if (H % 4 != 0):
            out = out[:, :, (4 - (H % 4)):H1, 0:W1]
            noise_map = noise_map[:, :, (4 - (H % 4)):H1, 0:W1]
        if (W % 4 != 0):
            out = out[:, :, 0:H, (4 - (W % 4)):W1]
            noise_map = noise_map[:, :, 0:H, (4 - (W % 4)):W1]
        out = out.view(N, 12, C, H, W)

        codebook = (torch.arange(12).float().to(x.device) + 1)

        # out = rb_decorrelatio(out)

        # out = tri_decorrelation_b(out)

        out = self.bd(out)

        out = (out - out.mean([2, 3, 4], keepdims=True)) / (out.std([2, 3, 4], keepdims=True) + 1e-8) * 5 / 255

        out = out * codebook.view(1, -1, 1, 1, 1)

        noise = out

        # cov_reg = (cov(noise) ** 2).sum() * H * W

        cov_reg = torch.zeros([]).to(x.device)

        pre_map = self.map(torch.cat([out.view(N, -1, H, W), noise_map], dim=1))

        pre_map_down = nn.AdaptiveAvgPool2d((H // 8, W // 8))(pre_map)

        mean_map = (pre_map_down.mean(1, keepdims=True) * 20).sigmoid() * 12

        # mean_map = noise_map * 255 / 10

        linear_x = 1 / ((codebook.view(1, -1, 1, 1) - mean_map) ** 2 + 1e-8)

        pre_map = linear_x.softmax(1)

        pre_map_up = F.interpolate(pre_map, (H, W))

        curr_map = (pre_map_up) * accu_map
        curr_map = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        map = pre_map_up.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map  = curr_map.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        map = map / (map.sum(1, keepdims=True) + 1e-5)

        # map = map.sqrt()

        x_mean = img.view(N, 1, C, H, W) - noise.view(N, 12, C, H, W)
        noise_fuse = noise.view(N, 12, C, H, W) * map.view(-1, 12, 1, 1, 1)
        noise_fuse = noise_fuse.sum(1)
        x = img - noise_fuse

        return x, curr_map.clamp(min=-1e3, max=1e3), x_mean




class RBDVDnet_Rec(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, mf2f=False):
        super(RBDVDnet_Rec, self).__init__()
        self.num_input_frames = num_input_frames
        self.pre_temp = RBBlock(num_input_frames=13, mf2f=mf2f, out=3, group=1, rep=True)

        self.temp = RBBlock(num_input_frames=1, mf2f=mf2f, out=36)
        self.map = nn.Conv2d(37, 12, kernel_size=3, stride=1, padding=1)
        self.reset_params()
        self.bd = BD(G=12)
        # self.buffers = torch.zeros([1, 3, 96, 96])

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, noise_map=None, accu_map=None, out_mean=None, clean=None):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''

        # hack
        N, C, H, W = img.shape

        if out_mean is None:
            out_mean = img.repeat(1, 12, 1, 1)
        x = torch.cat([img, out_mean], dim=1)
        if (H % 4 != 0):
            x = F.pad(x, [0, 0, 4 - (H % 4), 0], mode='reflect')
            noise_map = F.pad(noise_map, [0, 0, 4 - (H % 4), 0], mode='reflect')
        if (W % 4 != 0):
            x = F.pad(x, [4 - (W % 4), 0, 0, 0], mode='reflect')
            noise_map = F.pad(noise_map, [4 - (W % 4), 0, 0, 0], mode='reflect')

        noise_map_n = noise_map.repeat(1, 13, 1, 1)

        # Unpack inputs
        mid = self.pre_temp(x, noise_map_n)

        out = self.temp(mid, noise_map.repeat(1, 1, 1, 1))
        # unhack
        N1, C1, H1, W1 = out.shape
        if (H % 4 != 0):
            out = out[:, :, (4 - (H % 4)):H1, 0:W1]
            noise_map = noise_map[:, :, (4 - (H % 4)):H1, 0:W1]
        if (W % 4 != 0):
            out = out[:, :, 0:H, (4 - (W % 4)):W1]
            noise_map = noise_map[:, :, 0:H, (4 - (W % 4)):W1]
        out = out.view(N, 12, C, H, W)

        codebook = (torch.arange(12).float().to(x.device) + 1)

        # out = rb_decorrelatio(out)

        # out = tri_decorrelation_b(out)

        out = self.bd(out)

        out = (out - out.mean([2, 3, 4], keepdims=True)) / (out.std([2, 3, 4], keepdims=True) + 1e-8) * 5 / 255

        out = out * codebook.view(1, -1, 1, 1, 1)

        noise = out

        # cov_reg = (cov(noise) ** 2).sum() * H * W

        cov_reg = torch.zeros([]).to(x.device)

        noise_map = noise_map.mean(1, keepdim=True)
        pre_map = self.map(torch.cat([out.view(N, -1, H, W), noise_map], dim=1))

        pre_map_down = nn.AdaptiveAvgPool2d((H // 2, W // 2))(pre_map)

        mean_map = (pre_map_down.mean(1, keepdims=True) * 20).sigmoid() * 12

        # mean_map = noise_map * 255 / 10

        linear_x = 1 / (((codebook.view(1, -1, 1, 1) - mean_map).abs() + 1e-8) ** 0.5 + 1e-8)

        pre_map = linear_x.softmax(1)

        pre_map_up = F.interpolate(pre_map, (H, W))

        curr_map = pre_map_up * accu_map

        curr_map_out = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        # map = pre_map_up.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map  = curr_map.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map = map / (map.sum(1, keepdims=True) + 1e-5)

        # map = map.sqrt()

        map = curr_map_out

        x_mean = img.view(N, 1, C, H, W) - noise.view(N, 12, C, H, W)
        noise_fuse = noise.view(N, 12, C, H, W) * map.view(-1, 12, 1, H, W)

        # noise_corr = ( img.view(N, 1, C, H, W) - noise_fuse).sum(2)
        # curr_map = (curr_map) * noise_corr
        # curr_map = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        noise_fuse = noise_fuse.sum(1)
        x = img - noise_fuse
        # self.buf = noise_fuse.detach()

        return x, curr_map_out.clamp(min=-1e3, max=1e3), x_mean


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


class RBDVDnet_PostFuse(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, mf2f=False):
        super(RBDVDnet_PostFuse, self).__init__()
        self.num_input_frames = num_input_frames

        self.temp = RBBlock(num_input_frames=1, mf2f=mf2f, out=36)
        self.map = nn.Conv2d(37, 12, kernel_size=3, stride=1, padding=1)
        self.reset_params()
        self.bd = BD(G=12)
        self.bd2 = BD(G=12)

        self.norm = nn.BatchNorm2d(12)
        self.ref = Refine(in_ch=15)
        # self.buffers = torch.zeros([1, 3, 96, 96])

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, noise_map=None, accu_map=None, out_mean=None, clean=None):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''

        # hack
        N, C, H, W = img.shape

        x = img

        out_mean = img if out_mean is None else out_mean

        if (H % 4 != 0):
            x = F.pad(x, [0, 0, 4 - (H % 4), 0], mode='reflect')
            noise_map = F.pad(noise_map, [0, 0, 4 - (H % 4), 0], mode='reflect')
        if (W % 4 != 0):
            x = F.pad(x, [4 - (W % 4), 0, 0, 0], mode='reflect')
            noise_map = F.pad(noise_map, [4 - (W % 4), 0, 0, 0], mode='reflect')

        # Unpack inputs

        out = self.temp(x, noise_map.repeat(1, 1, 1, 1))
        # unhack
        N1, C1, H1, W1 = out.shape
        if (H % 4 != 0):
            out = out[:, :, (4 - (H % 4)):H1, 0:W1]
            noise_map = noise_map[:, :, (4 - (H % 4)):H1, 0:W1]
        if (W % 4 != 0):
            out = out[:, :, 0:H, (4 - (W % 4)):W1]
            noise_map = noise_map[:, :, 0:H, (4 - (W % 4)):W1]
        out = out.view(N, 12, C, H, W)

        codebook = (torch.arange(12).float().to(x.device) + 1)

        # out = rb_decorrelatio(out)

        # out = tri_decorrelation_b(out)

        out = self.bd(out)

        out = (out - out.mean([2, 3, 4], keepdims=True)) / (out.std([2, 3, 4], keepdims=True) + 1e-8) * 5 / 255

        out = out * codebook.view(1, -1, 1, 1, 1)

        noise = out

        noise_map = noise_map.mean(1, keepdim=True)
        pre_map = self.map(torch.cat([out.view(N, -1, H, W), noise_map], dim=1))

        pre_map_down = nn.AdaptiveAvgPool2d((H // 4, W // 4))(pre_map)

        mean_map = (pre_map_down.mean(1, keepdims=True) * 20).sigmoid() * 12

        # mean_map = noise_map * 255 / 10

        linear_x = 1 / (((codebook.view(1, -1, 1, 1) - mean_map).abs() + 1e-8) ** 0.5 + 1e-8)

        pre_map = linear_x.softmax(1)

        pre_map_up = F.interpolate(pre_map, (H, W))

        x_ref = torch.cat([pre_map_up, accu_map], dim=1)
        x_ref_w = (self.ref(x_ref) * 10).sigmoid()
        accu_map = x_ref_w * pre_map_up + (1 - x_ref_w) * accu_map

        curr_map = pre_map_up * accu_map

        curr_map_out = self.bd2(curr_map.view(-1, 12, 1, H, W))

        curr_map_out = self.norm(curr_map_out.view(N, -1, H, W))

        map = curr_map_out.softmax(1)

        # map = pre_map_up.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map  = curr_map.mean([2, 3]) * (accu_map * 1).mean([2, 3])

        # map = map / (map.sum(1, keepdims=True) + 1e-5)

        # map = map.sqrt()

        x_mean = img.view(N, 1, C, H, W) - noise.view(N, 12, C, H, W)
        noise_fuse = noise.view(N, 12, C, H, W) * map.view(-1, 12, 1, H, W)

        # noise_corr = ( img.view(N, 1, C, H, W) - noise_fuse).sum(2)
        # curr_map = (curr_map) * noise_corr
        # curr_map = curr_map / (curr_map.sum(1, keepdims=True) + 1e-5)

        noise_fuse = noise_fuse.sum(1)
        x = img - noise_fuse
        # self.buf = noise_fuse.detach()

        return x, curr_map_out.clamp(min=-1e3, max=1e3), x_mean


def cov_img(a, b):
    N, C, H, W = a.shape
    fuse = torch.cat([a.view(N, 1, C, H, W), b.view(N, 1, C, H, W)], dim=1)
    return cov_sum(fuse)
