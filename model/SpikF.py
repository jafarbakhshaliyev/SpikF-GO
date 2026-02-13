import torch
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class SPE(nn.Module):
    def __init__(self, input_len, patch_num, patch_dim, T, tau, D):
        super().__init__()
        self.patch_projector = nn.Linear(input_len // patch_num, patch_dim)
        self.bn = nn.BatchNorm2d(patch_dim)
        self.encoder_lif = MultiStepLIFNode(tau=tau, detach_reset=False, backend='torch')

        self.D = D
        self.T = T
        self.patch_dim = patch_dim
        self.patch_num = patch_num

    def forward(self, x):
        B, L, D = x.shape

        x = x.view(B, self.patch_num, L // self.patch_num, D).contiguous()
        x = x.transpose(-1, -2).contiguous()
        x = self.patch_projector(x)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.view(self.T, B, self.patch_dim, self.patch_num, D)
        x = self.encoder_lif(x)

        return x

class SFS(nn.Module):
    def __init__(self, patch_num, D, patch_dim, tau, alpha):
        super().__init__()
        self.time2freq = nn.Linear(patch_num, patch_num // 2 + 1)

        self.intra_conv = nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=[5, 1], stride=[1, 1], padding=[2, 0])
        self.inter_conv = nn.Conv2d(in_channels=patch_dim, out_channels=patch_dim, kernel_size=[3, 1], stride=[1, 1], padding=[1, 0])

        self.generator_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch', v_threshold=0.1)
        self.mp_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.sfs_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.intra_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.inter_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')

        self.bn1 = nn.BatchNorm2d(patch_dim)
        self.bn2 = nn.BatchNorm2d(patch_dim)
        self.bn3 = nn.BatchNorm2d(patch_dim)
        self.bn4 = nn.BatchNorm2d(patch_dim)

    def forward(self, x):
        res_x = x
        T, B, pd, pn, D = x.shape

        x = x.transpose(-1, -2).contiguous()
        freq_spec = torch.fft.rfft(x)

        selector = self.time2freq(x)
        selector = selector.flatten(0, 1)
        selector = self.bn1(selector)
        selector = selector.view(T, B, pd, D, -1)
        selector = self.generator_lif(selector)
        selector = selector.sum(dim=0, keepdim=True)
        selector = self.mp_lif(selector)
        selector = selector.repeat(T, 1, 1, 1, 1).float()
        selector_imag = torch.zeros(selector.size()).to(x.device)
        selector = torch.complex(selector, selector_imag).to(x.device)

        remain_freq = selector * freq_spec

        current = torch.fft.irfft(remain_freq)
        current = current.transpose(-1, -2).contiguous()
        current = current.flatten(0, 1)
        current = self.bn2(current)
        current = current.view(T, B, pd, pn, D)

        spike = self.sfs_lif(current)
        x = spike + res_x
        res_x = x

        x = x.flatten(0, 1)
        x = self.intra_conv(x)
        x = self.bn3(x)
        x = x.view(T, B, pd, pn, D)
        x = self.intra_lif(x) + res_x
        res_x = x

        x = x.transpose(0, 3).contiguous()
        x = x.flatten(0, 1)
        x = self.inter_conv(x)
        x = self.bn4(x)
        x = x.view(pn, B, pd, T, D)
        x = x.transpose(0, 3)
        x = self.inter_lif(x)
        x = x + res_x
        
        return x

class SpikF(nn.Module):
    def __init__(self, args, input_len, patch_num, patch_dim, T, blocks, D, pred_len, tau, alpha, hidden_dim):
        super().__init__()
        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)
        self.args = args

        self.SFSs = nn.ModuleList()
        for i in range(blocks):
            self.SFSs.append(SFS(patch_num, D, patch_dim, tau, alpha))

        self.dense1 = nn.Linear(patch_num * patch_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)

        self.bn = nn.BatchNorm1d(D)

        self.activ = nn.GELU()

    def forward(self, x):

        if self.args.normalize:

            mean = x.mean(dim=1, keepdim=True).detach()
            x = x - mean

            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / std

        x = self.SPE(x)
        T, B, pd, pn, D = x.shape

        for i in range(len(self.SFSs)):
            x = self.SFSs[i](x)
            
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(-2, -1)
        x = self.dense1(x)
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(T, B, -1, D)

        if self.args.normalize:
            x = x * std
            x = x + mean.repeat(T, 1, 1, 1)

        aux = {
                 # <— NEW
            'gate_l0': torch.tensor(0.0, device=x.device)  # Placeholder for compatibility
        }
        

        return x, aux
