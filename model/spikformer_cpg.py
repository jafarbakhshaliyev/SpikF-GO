from typing import Optional

from pathlib import Path
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

import math
from dataclasses import dataclass
import warnings



tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


@dataclass
class CPG(nn.Module):
    num_neurons: int = 40
    w_max: float = 10000.0
    l_max: int = 5000

    def __post_init__(self):
        self._cpg = torch.zeros(self.l_max, self.num_neurons)
        position = torch.arange(0, self.l_max, dtype=torch.float).unsqueeze(
            1
        )  # MaxL, 1
        div_term = torch.exp(
            torch.arange(0, self.num_neurons, 2).float()
            * (-math.log(self.w_max) / self.num_neurons)
        )
        div_term_single = torch.exp(
            torch.arange(0, self.num_neurons - 1, 2).float()
            * (-math.log(self.w_max) / self.num_neurons)
        )
        self._cpg[:, 0::2] = torch.heaviside(
            torch.sin(position * div_term) - 0.8, torch.tensor([1.0])
        )
        self._cpg[:, 1::2] = torch.heaviside(
            torch.cos(position * div_term_single) - 0.8, torch.tensor([1.0])
        )

    @property
    def cpg(self):
        return self._cpg


class CPGLinear(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, cpg: CPG = CPG(), dropout: float = 0.1
    ):
        super().__init__()
        self.cpg = nn.Parameter(cpg.cpg, requires_grad=False)
        self.inp_linear = nn.Linear(input_size, output_size)
        self.cpg_linear = nn.Linear(cpg.num_neurons, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # B TL D
        cpg = self.cpg[: x.size(-2)]
        x = self.dropout(x)
        return self.inp_linear(x) + self.cpg_linear(cpg)


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # T B L C
        inputs = inputs.permute(0, 1, 3, 2)  # T B C L
        spks = self.lif(inputs)  # T B C L
        return spks


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # B, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # B, C, L, 1
        enc = self.enc(delta)  # B, C, L, T
        enc = enc.permute(3, 0, 1, 2)  # T, B, C, L
        spks = self.lif(enc)
        return spks


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, C, L
        enc = self.encoder(inputs)  # B, T, C, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, C, L
        spks = self.lif(enc)  # T, B, C, L
        return spks



SpikeEncoder = {
    "snntorch": {
        "repeat": RepeatEncoder,
        "conv": ConvEncoder,
        "delta": DeltaEncoder,
    },
    "spikingjelly": {
        "repeat": RepeatEncoder,
        "conv": ConvEncoder,
        "delta": DeltaEncoder,
    },
}





class SSA(nn.Module):
    def __init__(
        self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.attn_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr / 2,
            backend=backend,
        )

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

    def forward(self, x):
        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB L D
        q_m_out = self.q_m(x_for_qkv)  # TB L D
        q_m_out = (
            self.q_bn(q_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        q_m_out = self.q_lif(q_m_out)
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = (
            self.k_bn(k_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        k_m_out = self.k_lif(k_m_out)
        k = (
            k_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = (
            self.v_bn(v_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        v_m_out = self.v_lif(v_m_out)
        v = (
            v_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        x = attn @ v  # x_shape: T * B * heads * L * D//heads

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x


class MLP(nn.Module):
    def __init__(
        self,
        length,
        tau,
        common_thr,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = CPGLinear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.fc2 = CPGLinear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.transpose(0, 1).flatten(1, 2)  # B TL D
        x = self.fc1(x)  # B TL H
        x = (
            self.bn1(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, self.hidden_features)
            .contiguous()
        )  # B T L H
        x = self.lif1(x.transpose(0, 1)).transpose(0, 1)  # B T L H
        x = x.flatten(1, 2)  # B TL H
        x = self.fc2(x)  # B TL D
        x = (
            self.bn2(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, D)
            .contiguous()
        )  # B T L D
        x = self.lif2(x.transpose(0, 1))  # T B L D
        return x


class Block(nn.Module):
    def __init__(
        self,
        length,
        tau,
        common_thr,
        dim,
        d_ff,
        heads=8,
        qkv_bias=False,
        qk_scale=0.125,
    ):
        super().__init__()
        self.attn = SSA(
            length=length,
            tau=tau,
            common_thr=common_thr,
            dim=dim,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.mlp = MLP(
            length=length,
            tau=tau,
            common_thr=common_thr,
            in_features=dim,
            hidden_features=d_ff,
        )

    def forward(self, x):
        # T B L D
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikformerCPG(nn.Module):
    def __init__(
        self,
        args,
        dim: int=256,
        d_ff: Optional[int] = None,
        num_pe_neuron: int = 40,
        pe_type: str = "neuron",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 10000.0,  # "100" or "1000" or "10000"
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 5000,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        super().__init__()
        self.dim = 256
        self.d_ff = 1024
        self.T = args.T
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.input_size = args.feature_size
        self.pre_length = args.pre_length
        self.args = args


        self._snn_backend = "spikingjelly"

        self.temporal_encoder = SpikeEncoder[self._snn_backend]["conv"](num_steps)
        self.encoder = CPGLinear(self.input_size, dim, CPG(num_neurons=num_pe_neuron))

        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for _ in range(depths)
            ]
        )

        self.apply(self._init_weights)

        self.fc = nn.Linear(args.seq_length*dim, args.pre_length*args.feature_size)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)

        if self.args.normalize:

            mean = x.mean(dim=1, keepdim=True).detach() # shape [B, 1, D]
            x = x - mean

            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / std


        x = self.temporal_encoder(x)  # B L C -> T B C L
        T, B, _, L = x.shape
        x = x.permute(1, 0, 3, 2)  # B T L C
        x = x.flatten(1, 2)  # B TL C
        x = self.encoder(x)  # B TL D
        x = x.reshape(B, T, L, -1).permute(1, 0, 2, 3)  # T B L D
        x = self.init_lif(x)

        for blk in self.blocks:
            x = blk(x)  # T B L D
        out = x.mean(0)
        out = self.fc(out.flatten(-2, -1)).reshape(-1, self.pre_length, self.input_size)  # B D L -> B L D
        #print(out.shape)
        if self.args.normalize:
            out = out * std + mean  # denormalization
        aux = {'gate_l0': torch.tensor(0.0, device=out.device)}
        return out, aux  # B D L -> B L D

