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

class iSSA(nn.Module):
    def __init__(self, patch_num, D, patch_dim, tau, alpha):
        super().__init__()
        self.lin1 = nn.Linear(patch_num, patch_num)
        self.lin2 = nn.Linear(patch_num, patch_num)
        self.lin3 = nn.Linear(patch_num, patch_num)

        self.lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.lif3 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.lif4 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')

        self.b1 = nn.BatchNorm2d(patch_dim)
        self.b2 = nn.BatchNorm2d(patch_dim)
        self.b3 = nn.BatchNorm2d(patch_dim)
        self.b4 = nn.BatchNorm2d(patch_dim)
     
    def forward(self, x):
        res_x = x
        T, B, pd, pn, D = x.shape


        x = x.transpose(-1, -2).contiguous()
        q = self.lin1(x).flatten(0, 1)
        k = self.lin2(x).flatten(0, 1)
        v = self.lin3(x).flatten(0, 1)

        q = self.b1(q)
        k = self.b2(k)
        v = self.b3(v)

        q = q.view(T, B, pd, D, -1)
        k = k.view(T, B, pd, D, -1)
        v = v.view(T, B, pd, D, -1)

        q = self.lif1(q)
        k = self.lif2(k).transpose(-1, -2).contiguous()
        v = self.lif3(v)

        attn = q @ k
        attn = attn @ v
        attn = attn.flatten(0, 1)
        attn = self.b4(attn)
        attn = attn.view(T, B, pd, D, pn)
        attn = self.lif4(attn)
        attn = attn.transpose(-1, -2).contiguous()

        return attn

class iSpikformer(nn.Module):
    def __init__(self, args, input_len, patch_num, patch_dim, T, blocks, D, pred_len, tau, alpha, hidden_dim):
        super().__init__()
        self.emb = SPE(input_len, patch_num, patch_dim, T, tau, D)
        self.args = args
        self.attn = nn.ModuleList()
        for i in range(blocks):
            self.attn.append(iSSA(patch_num, D, patch_dim, tau, alpha))

        self.dense1 = nn.Linear(patch_num*patch_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)
        self.bn = nn.BatchNorm1d(D)
        self.activ = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.to('cuda:0') 


    def forward(self, x):
        if self.args.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()
            x = x - mean

            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / std

        x = self.emb(x)
        T, B, pd, pn, D = x.shape

        for i in range(len(self.attn)):
            x = self.attn[i](x)
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
            'gate_l0': torch.tensor(0.0, device=x.device)  # placeholder
        }
        
        return x.mean(dim=0), aux
    