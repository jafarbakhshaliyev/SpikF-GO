from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import copy
import math

def generate_ones_and_minus_ones_matrix(rows, cols):
    random_matrix = torch.randint(0, 2, (rows, cols))
    binary_matrix = torch.where(
        random_matrix == 0,
        -1 * torch.ones_like(random_matrix),
        torch.ones_like(random_matrix),
    )
    return binary_matrix.float()


class RandomPE(nn.Module):
    def __init__(
        self,
        d_model,
        pe_mode="concat",
        num_pe_neuron=10,
        neuron_pe_scale=1000.0,
        dropout=0.1,
        num_steps=4,
    ):
        super().__init__()
        self.max_len = 5000  # different from windows
        self.pe_mode = pe_mode
        self.neuron_pe_scale = neuron_pe_scale
        self.dropout = nn.Dropout(p=dropout)
        if self.pe_mode == "concat":
            self.num_pe_neuron = copy.deepcopy(num_pe_neuron)
        elif self.pe_mode == "add":
            self.num_pe_neuron = copy.deepcopy(d_model)
        pe = generate_ones_and_minus_ones_matrix(
            self.max_len, self.num_pe_neuron
        )  # MaxL, Neur
        pe = pe.unsqueeze(0).transpose(0, 1)  # MaxL, 1, Neur
        print("pe.shape: ", pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # # L, TB, D
        # if self.pe_mode == "concat":
        #     x = torch.concat([x, self.pe[: x.size(0), :].repeat(1, x.size(1), 1)], dim=-1)
        #     # print(x.shape) # L TB D'
        # elif self.pe_mode == "add":
        #     x = x + self.pe[:x.size(0), :]
        # x = self.dropout(x)

        # T, B, L, D
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3)  # B, T, L, D
        x = x.flatten(1, 2)  # B, TL, D
        if self.pe_mode == "concat":
            # tmp: TL, 1, Neur -> TL, B, Neur -> B, TL, Neur
            tmp = self.pe[: x.size(-2), :].repeat(1, B, 1).transpose(0, 1)
            x = torch.concat([x, tmp], dim=-1)
            # print(x.shape) # B, TL, D'
        elif self.pe_mode == "add":
            # [B, TL, D] + [1, TL, Neur]
            x = x + self.pe[: x.size(-2), :].transpose(0, 1)
            # print(x.shape) # B, TL, D
        x = x.transpose(0, 1)  # TL, B D
        x = x.reshape(T, L, B, -1)  # T, L, B, D
        x = x.permute(0, 2, 1, 3)  # T, B, L, D
        return self.dropout(x)


class NeuronPE(nn.Module):
    def __init__(
        self,
        d_model,
        pe_mode="concat",
        num_pe_neuron=10,
        neuron_pe_scale=10000.0,
        dropout=0.1,
        num_steps=4,
    ):
        super().__init__()
        self.max_len = 50000  # different from windows
        self.pe_mode = pe_mode
        self.neuron_pe_scale = neuron_pe_scale
        self.dropout = nn.Dropout(p=dropout)
        if self.pe_mode == "concat":
            self.num_pe_neuron = copy.deepcopy(num_pe_neuron)
        elif self.pe_mode == "add":
            self.num_pe_neuron = copy.deepcopy(d_model)
        pe = torch.zeros(self.max_len, self.num_pe_neuron)  # MaxL, Neur
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(
            1
        )  # MaxL, 1
        div_term = torch.exp(
            torch.arange(0, self.num_pe_neuron, 2).float()
            * (-math.log(neuron_pe_scale) / self.num_pe_neuron)
        )
        div_term_single = torch.exp(
            torch.arange(0, self.num_pe_neuron - 1, 2).float()
            * (-math.log(neuron_pe_scale) / self.num_pe_neuron)
        )
        pe[:, 0::2] = torch.heaviside(
            torch.sin(position * div_term) - 0.8, torch.tensor([1.0])
        )
        pe[:, 1::2] = torch.heaviside(
            torch.cos(position * div_term_single) - 0.8, torch.tensor([1.0])
        )
        pe = pe.unsqueeze(0).transpose(0, 1)  # MaxL, 1, Neur
        print("pe.shape: ", pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # # L, TB, D
        # if self.pe_mode == "concat":
        #     x = torch.concat([x, self.pe[: x.size(0), :].repeat(1, x.size(1), 1)], dim=-1)
        #     # print(x.shape) # L TB D'
        # elif self.pe_mode == "add":
        #     x = x + self.pe[:x.size(0), :]
        # x = self.dropout(x)

        # T, B, L, D
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3)  # B, T, L, D
        x = x.flatten(1, 2)  # B, TL, D
        if self.pe_mode == "concat":
            # tmp: TL, 1, Neur -> TL, B, Neur -> B, TL, Neur
            tmp = self.pe[: x.size(-2), :].repeat(1, B, 1).transpose(0, 1)
            x = torch.concat([x, tmp], dim=-1)
            # print(x.shape) # B, TL, D'
        elif self.pe_mode == "add":
            # [B, TL, D] + [1, TL, Neur]
            # print(self.pe[:x.size(-2), :].shape)
            x = x + self.pe[: x.size(-2), :].transpose(0, 1)
            # print(x.shape) # B, TL, D
        x = x.transpose(0, 1)  # TL, B D
        x = x.reshape(T, L, B, -1)  # T, L, B, D
        x = x.permute(0, 2, 1, 3)  # T, B, L, D
        return self.dropout(x)


class StaticPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # MaxL, D
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # MaxL, 1
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        div_term_single = torch.exp(
            torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: L, TB, D
        x = x + self.pe[: x.size(0), :]
        x = self.dropout(x)
        return x


class ConvPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, num_steps=4):

        super().__init__()
        self.T = num_steps
        self.rpe_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm1d(d_model)
        self.rpe_lif = neuron.LIFNode(
            step_mode="m",
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
            v_threshold=1.0,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: L, TB, D
        L, TB, D = x.shape
        x_feat = x.permute(1, 2, 0)  # TB, D, L
        x_feat = self.rpe_conv(x_feat)  # TB, D, L
        x_feat = (
            self.rpe_bn(x_feat).reshape(self.T, int(TB / self.T), D, L).contiguous()
        )  # T, B, D, L
        x_feat = self.rpe_lif(x_feat)
        x_feat = x_feat.flatten(0, 1)  # TB, D, L
        x_feat = self.dropout(x_feat)  # TB, D, L
        x_feat = x_feat.permute(2, 0, 1)  # L, TB, D
        x = x + x_feat
        return x


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        pe_type: str,
        max_len: int = 5000,
        pe_mode: str = "add",
        num_pe_neuron: int = 10,
        neuron_pe_scale: float = 1000.0,
        dropout=0.1,
        num_steps=4,
    ):
        super().__init__()
        self.emb_type = pe_type
        if pe_type in ["learn", "none"]:
            self.emb = nn.Embedding(max_len, input_size)
        elif pe_type == "conv":
            self.emb = ConvPE(
                d_model=input_size,
                max_len=max_len,
                dropout=dropout,
                num_steps=num_steps,
            )
        elif pe_type == "static":
            self.emb = StaticPE(d_model=input_size, max_len=max_len, dropout=dropout)
        elif pe_type == "neuron":
            self.emb = NeuronPE(
                d_model=input_size,
                pe_mode=pe_mode,
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                dropout=dropout,
                num_steps=num_steps,
            )
        elif pe_type == "random":
            self.emb = RandomPE(
                d_model=input_size,
                pe_mode=pe_mode,
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                dropout=dropout,
                num_steps=num_steps,
            )
        else:
            raise ValueError("Unknown embedding type: {}".format(pe_type))

    def forward(self, x):
        if self.emb_type == "learn":
            # T, B, L, D = x.shape # x: T, B, L, D
            # x = x.flatten(0, 1) # TB, L, D
            tmp = torch.arange(
                end=x.size()[1], device=x.device
            )  # [0,1,2,...,L-1], shape: L
            embedding = self.emb(tmp)  # shape: L, D
            embedding = embedding.repeat([x.size()[0], 1, 1])  # TB, L, D'
            x = x + embedding
            # x = x.reshape(T, B, L, -1)
        elif self.emb_type in ["static", "conv"]:
            T, B, L, _ = x.shape  # x: T, B, L, D
            x = x.flatten(0, 1)  # TB, L, D
            x = self.emb(x.transpose(0, 1)).transpose(0, 1)  # x: TB, L, D'
            x = x.reshape(T, B, L, -1)
        elif self.emb_type in ["neuron", "random"]:
            T, B, L, _ = x.shape  # x: T, B, L, D
            # T, B, L, D
            x = self.emb(x)
            x = x.reshape(T, B, L, -1)
        return x  # T, B, L, D'






class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # out_size batch L C
        inputs = inputs.permute(1, 0, 3, 2)  # batch out_size L C
        spks = self.lif(inputs)
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
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=False,
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # batch, 1, C, L
        enc = self.encoder(inputs)  # batch, output_size, C, L
        spks = self.lif(enc)
        return spks


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # batch, C, L, 1
        enc = self.enc(delta)  # batch, C, L, output_size
        enc = enc.permute(0, 3, 1, 2)  # batch, output_size, C, L
        spks = self.lif(enc)
        return spks



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, : -self.chomp_size].contiguous()



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



class SpikeTemporalBlock2D(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        num_steps=4,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=(1, dilation),
            )
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.chomp1 = Chomp2d(padding)
        self.lif1 = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=(1, dilation),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.chomp2 = Chomp2d(padding)
        self.lif2 = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        )
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out1 = self.chomp1(self.bn1(self.conv1(x)))
        spk_rec1 = []
        for _ in range(self.num_steps):
            spk = self.lif1(out1)
            spk_rec1.append(spk)
        spks1 = torch.stack(spk_rec1, dim=-1)  # spks1: B, H, C, L, T
        spks1 = spks1.mean(-1)  # spks1: B, H, C, L

        out2 = self.chomp2(self.bn2(self.conv2(spks1)))
        spk_rec2 = []
        for _ in range(self.num_steps):
            spk = self.lif2(out2)
            spk_rec2.append(spk)
        spks2 = torch.stack(spk_rec2, dim=-1)  # spks2: B, H, C, L, T
        spks2 = spks2.mean(-1)  # spks2: B, H, C, L

        if torch.isnan(spks2).any() or torch.isinf(spks2).any():
            print("illegal value in TemporalBlock2D")

        if self.downsample is None:
            res = x
        else:
            res = self.downsample(x)
        spk_rec3 = []
        for _ in range(self.num_steps):
            spk = self.lif(spks2 + res)
            spk_rec3.append(spk)

        res = torch.stack(spk_rec3, dim=-1)  # res: B, H, C, L, T
        res = res.mean(-1)

        return res


class SpikeTCN(nn.Module):
    

    def __init__(
        self,
        args,
        num_levels: int=3,
        channel: int=16,
        dilation: int=2,
        stride: int = 1,
        num_steps: int = 16,
        kernel_size: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 40,
        pe_type: str = "neuron",
        pe_mode: str = "concat",  # "add" or "concat"
        neuron_pe_scale: float = 10000.0,  # "100" or "1000" or "10000"
    ):
        """
        Args:
            num_channels: The number of convolutional channels in each layer.
            kernel_size: The kernel size of convolutional layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.pe_type = pe_type
        self._snn_backend = "snntorch"
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.hidden_size   = args.hidden_size  
        self.num_steps = args.T    
        self.input_size = args.feature_size  
        self.pre_length = args.pre_length      
        self.num_levels = args.blocks
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.kernel_size = args.kernel_size

        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](self.hidden_size)
        self.args = args

 
        self.pe = PositionEmbedding(
            pe_type=pe_type,
            pe_mode=pe_mode,
            neuron_pe_scale=neuron_pe_scale,
            input_size=self.input_size,
            max_len=max_length,
            num_pe_neuron=self.num_pe_neuron,
            dropout=0.1,
            num_steps=self.num_steps,
        )
        layers = []
        num_channels = [channel] * self.num_levels
        num_channels.append(1)
        for i in range(self.num_levels + 1):
            dilation_size = dilation**i
            in_channels = self.hidden_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                SpikeTemporalBlock2D(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    stride=stride,
                    dilation=dilation_size,
                    padding=(self.kernel_size - 1) * dilation_size,
                    num_steps=self.num_steps,
                )
            ]

        self.network = nn.Sequential(*layers)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.__output_size = args.feature_size + num_pe_neuron
        else:
            self.__output_size =  args.seq_length

        self.fc1 = nn.Linear(self.__output_size, args.feature_size)
        self.fc2 = nn.Linear(args.seq_length, self.pre_length)
        self.to('cuda:0') 

    def forward(self, inputs: torch.Tensor):
        utils.reset(self.encoder)
        for layer in self.network:
            utils.reset(layer)
        
        if self.args.normalize:
            mean = inputs.mean(dim=1, keepdim=True).detach() # shape [B, 1, D]
            inputs = inputs - mean

            std = torch.sqrt(torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            inputs = inputs / std

        inputs = self.encoder(inputs)  # B, H, C, L
        if self.pe_type != "none":
            # B, H, C, L -> H B L C' -> B H C' L
            inputs = self.pe(inputs.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
        spks = self.network(inputs)
        spks = spks.squeeze(1)  # B, C', L

        preds = self.fc1(spks.permute(0, 2, 1))  # B, L, C
        preds = self.fc2(preds.permute(0, 2, 1))  # B, C', L
        #.squeeze(-1) # B, O, C'
        preds = preds.permute(0, 2, 1).contiguous()
        if self.args.normalize:
            preds = preds * std + mean  # denormalize
        aux = {'gate_l0': torch.tensor(0.0, device=preds.device)}
        
        return preds, aux

    @property
    def output_size(self):
        return self.__output_size


