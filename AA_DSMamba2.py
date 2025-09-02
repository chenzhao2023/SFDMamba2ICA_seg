import torch
from torch import nn
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from abc import abstractmethod

class AA_DSMamba2(nn.Module):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__()
        self.DS_mamba2_H = DSMamba2_1d(cout, cout, d_model, **mamba2_args)
        self.DS_mamba2_W = DSMamba2_1d(cin, cout, d_model, **mamba2_args)
        self.Conv = nn.Conv2d(cout * 2, cout, 1)

    def forward(self, x):
        # 非对称卷积
        b, c, h, w = x.shape
        # 先 W 再 H
        x_w1 = x.transpose(1, 2).reshape(b * h, c, w)  # bh, c, w
        y_w1 = self.DS_mamba2_W(x_w1).reshape(b, h, -1, w).transpose(1, 2)
        x_h1 = y_w1.transpose(1, 3).reshape(b * w, h, -1).transpose(1, 2)  # bw, c, h
        y1 = self.DS_mamba2_H(x_h1).transpose(1, 2).reshape(b, w, h, -1).transpose(1, 3)

        # 先 H 再 W
        x_h2 = x.transpose(1, 3).reshape(b * w, h, c).transpose(1, 2)  # bw, c, h
        y_h2 = self.DS_mamba2_W(x_h2).transpose(1, 2).reshape(b, w, h, -1).transpose(1, 3)
        x_w2 = y_h2.transpose(1, 2).reshape(b * h, -1, w)  # bh, c, w
        y2 = self.DS_mamba2_H(x_w2).reshape(b, h, -1, w).transpose(1, 2)
        # 合并结果
        y = self.Conv(torch.cat([y1, y2], dim=1))
        return y

class DS_Mamba2(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.in_C = nn.Linear(cin, d_model, bias=False)  # project input channels to d_model
        self.mamba2_for = mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, ) # forward direction
        self.mamba2_back = mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, ) # backward direction
        self.out_C = nn.Linear(d_model, cout, bias=False)  # project d_model back to output channels
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(self, x):
        pass


class DSMamba2_1d(DS_Mamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # pad length l to a multiple of 4: [B, C64, L4]
        x = x.transpose(1, 2)   # convert to 1-D signal: [B, C64, D4*W4*H4]
        x = self.in_C(x)  # project to target channel dimension
        x = self.mamba2_for(x) + self.mamba2_back(x.flip(1)).flip(1)
        x = self.out_C(x)  # project back to target channel dimension
        x = x.transpose(1, 2)  # convert back to 1-D signal: [B, C64, D4*W4*H4]
        x = x[:, :, :l]  # crop to original length
        return x

class mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        # self.chunk_size = torch.tensor(chunk_size, dtype=torch.int32)
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by self.headdim"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, )
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMS(self.d_inner, )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, )

    def forward(self, u: Tensor):
        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        def silu(x): return x * F.sigmoid(x)
        # Pad or truncate xBC seqlen to d_conv
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        y = self.SSD(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2), )

        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.norm(y, z)
        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def SSD(self, x, A, B, C):
        chunk_size = self.chunk_size
        # if x.shape[1] % chunk_size == 0:
        #
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        # Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )

        return Y

class RMS(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        def silu(x): return x * F.sigmoid(x)
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


if __name__ == '__main__':
    net = AA_DSMamba2(1024, 128, 32).cuda()
    x = torch.randn(2, 1024, 32, 32).cuda()
    print(net(x).shape)