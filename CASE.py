import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CASE(nn.Module):
    def __init__(self, scales=[1, 2, 3], beta=0.5, c=15):
        """
        scales: Gaussian smoothing sigma list
        beta, c: parameters (default tuned for visualization)
        amplify_factor: value to multiply vesselness before clamping to [0, 1]
        """
        super().__init__()
        self.scales = scales
        self.beta = beta
        self.c = c
        #self.amplify_factor = amplify_factor

        # Fixed Hessian kernels
        kernel_xx = torch.tensor([[1, -2, 1],
                                  [2, -4, 2],
                                  [1, -2, 1]], dtype=torch.float32) / 4
        kernel_yy = kernel_xx.T
        kernel_xy = torch.tensor([[1, 0, -1],
                                  [0, 0, 0],
                                  [-1, 0, 1]], dtype=torch.float32) / 4

        self.register_buffer("kernel_xx", kernel_xx[None, None])
        self.register_buffer("kernel_yy", kernel_yy[None, None])
        self.register_buffer("kernel_xy", kernel_xy[None, None])

    def gaussian_blur(self, x, sigma):
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        padding = kernel_size // 2
        t = torch.arange(-padding, padding + 1, dtype=torch.float32, device=x.device)
        gauss = torch.exp(-0.5 * (t / sigma) ** 2)
        gauss = gauss / gauss.sum()
        gauss_x = gauss.view(1, 1, 1, -1)
        gauss_y = gauss.view(1, 1, -1, 1)
        C = x.shape[1]
        x = F.conv2d(x, gauss_x.expand(C, 1, 1, -1), padding=(0, padding), groups=C)
        x = F.conv2d(x, gauss_y.expand(C, 1, -1, 1), padding=(padding, 0), groups=C)
        return x

    def compute_frangi(self, x):
        ch = x[:, 0:1, :, :]  # for grayscale
        Dxx = F.conv2d(ch, self.kernel_xx, padding=1)
        Dyy = F.conv2d(ch, self.kernel_yy, padding=1)
        Dxy = F.conv2d(ch, self.kernel_xy, padding=1)

        tmp = torch.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2 + 1e-12)
        lambda1 = 0.5 * (Dxx + Dyy + tmp)
        lambda2 = 0.5 * (Dxx + Dyy - tmp)

        # Use absolute values and sort eigenvalues
        abs_lambda1 = torch.abs(lambda1)
        abs_lambda2 = torch.abs(lambda2)
        lambda1, lambda2 = torch.max(abs_lambda1, abs_lambda2), torch.min(abs_lambda1, abs_lambda2)

        # Correct Rb definition
        rb = lambda2 / (lambda1 + 1e-12)
        s = torch.sqrt(lambda1**2 + lambda2**2 + 1e-12)

        # Compute vesselness
        term1 = torch.exp(-rb**2 / (2 * self.beta**2))
        term2 = 1 - torch.exp(-s**2 / (2 * self.c**2))
        vesselness = term1 * term2

        # Amplify for visualization
        #vesselness = vesselness * self.amplify_factor
        vesselness = vesselness.clamp(0, 1)

        return vesselness

    def forward(self, x):
        responses = []
        for sigma in self.scales:
            blurred = self.gaussian_blur(x, sigma)
            vesselness = self.compute_frangi(blurred)
            responses.append(vesselness)
        stacked = torch.stack(responses, dim=0)
        return torch.max(stacked, dim=0)[0]  # Final Vesselness

if __name__ == '__main__':
    block = CASE()
    input = torch.rand(2, 1, 512, 512)
    output = block(input)
    print("Input shape:", input.size())
    print("Output shape:", output.size())