import pywt
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

def Wavelet_Filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def WT(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def IWT(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class wt(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.save_for_backward(filters)
        x = WT(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        filters, = ctx.saved_tensors
        grad_input = IWT(grad_output, filters)
        return grad_input, None

class iwt(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.save_for_backward(filters)
        x = IWT(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        filters, = ctx.saved_tensors
        grad_input = WT(grad_output, filters)
        return grad_input, None

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)

class PHFP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(PHFP, self).__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        self.wt_filter, self.iwt_filter = Wavelet_Filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=stride, bias=bias, groups=in_channels)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1,
                       dilation=1, groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_x = wt.apply(curr_x_ll, self.wt_filter)
            x_ll_in_levels.append(curr_x[:, :, 0, :, :])
            x_h_in_levels.append(curr_x[:, :, 1:4, :, :])
            curr_x_ll = curr_x[:, :, 0, :, :]

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = iwt.apply(curr_x, self.iwt_filter)

        x_tag = next_x_ll
        x = self.base_conv(x)
        x = self.base_scale(x)
        x = x + x_tag
        return x

if __name__ == '__main__':
    block = PHFP(in_channels=1, out_channels=1)
    input = torch.rand(2, 1, 512, 512)
    output = block(input)
    print("Input shape:", input.size())
    print("Output shape:", output.size())