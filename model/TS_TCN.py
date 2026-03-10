from typing import Optional, Callable

import torch
from torch import nn
from torch.nn.utils import weight_norm
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import numpy as np
import math
import copy
from spikingjelly.activation_based import surrogate, neuron
from abc import abstractmethod
import warnings



surrogate.atan = lambda alpha=2.0: SG.apply


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


@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):

    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None
             

class SG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class MemoryModule(nn.Module):
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _MemoryModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _MemoryModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _MemoryModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        """
        * :ref:`API in English <MemoryModule.memories-en>`

        .. _MemoryModule.memories-cn:

        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.memories-cn>`

        .. _MemoryModule.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`API in English <MemoryModule.named_memories-en>`

        .. _MemoryModule.named_memories-cn:

        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.named_memories-cn>`

        .. _MemoryModule.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """

        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _MemoryModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _MemoryModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class StepModule:
    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`
        .. _StepModule.supported_step_mode-cn:
        :return: 包含支持的后端的tuple
        :rtype: tuple[str]
        返回此模块支持的步进模式。
        * :ref:`中文 API <StepModule.supported_step_mode-cn>`
        .. _StepModule.supported_step_mode-en:
        :return: a tuple that contains the supported backends
        :rtype: tuple[str]
        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`
        .. _StepModule.step_mode-cn:
        :return: 模块当前使用的步进模式
        :rtype: str
        * :ref:`中文 API <StepModule.step_mode-cn>`
        .. _StepModule.step_mode-en:
        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`
        .. _StepModule.step_mode-setter-cn:
        :param value: 步进模式
        :type value: str
        将本模块的步进模式设置为 ``value``
        * :ref:`中文 API <StepModule.step_mode-setter-cn>`
        .. _StepModule.step_mode-setter-en:
        :param value: the step mode
        :type value: str
        Set the step mode of this module to be ``value``
        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value



class BaseNode(MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = True):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('v_s', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq


        self.alpha_s = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float)) 
        self.alpha_l = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float))

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v


    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold, 2.0)

    def sl_neuronal_fire(self):
        s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
        return s_s, s_l

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        s_s, s_l = self.sl_neuronal_fire()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
        self.neuronal_reset(s_s, s_l)
        
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):

        T = x_seq.shape[-1]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:, t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        outputs = torch.stack(y_seq, dim=0).permute(1, 0)

        return outputs

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TSLIFNode(BaseNode):
    def __init__(self,
                 v_threshold=1.0,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
                 gamma: float = 0.5):
        super(TSLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)
        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
        self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']
        self.v = self.names['v2']
        self.v_s = self.names['v1']

    def neuronal_reset(self, spike_s, spike_l):
        if not self.hard_reset:
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l, self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
        else:
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_s, self.v_reset)

    def forward(self, x: torch.Tensor):
        return super().single_step_forward(x)
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"





class SpikeTemporalBlock(nn.Module):
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

        self.tslif1 = TSLIFNode(
            surrogate_function =SG.apply,
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

        self.tslif2 = TSLIFNode(
            surrogate_function =SG.apply,
        )

        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        )

        self.tslif = TSLIFNode(
            surrogate_function =SG.apply,
        )

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # out1: 24, 16, 361, 168
        
        out1 = self.chomp1(self.bn1(self.conv1(x)))
        spk_rec1 = []
        for _ in range(self.num_steps):
            spk = self.tslif1(out1)
            spk_rec1.append(spk)

        spks1 = torch.stack(spk_rec1, dim=-1)  # spks1: B, H, C, L, T
        spks1 = spks1.mean(-1)  # spks1: B, H, C, L

        out2 = self.chomp2(self.bn2(self.conv2(spks1)))
        spk_rec2 = []
        for _ in range(self.num_steps):
            # spk: 24, 16, 361, 168
            spk = self.tslif2(out2)
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

            spk = self.tslif(spks2 + res)
            spk_rec3.append(spk)


        res = torch.stack(spk_rec3, dim=-1)  # res: B, H, C, L, T

        res = res.mean(-1)

        return res





class TSTCN(nn.Module):
    def __init__(
        self,
        args,
        num_levels: int = 3,
        channel: int = 16,
        dilation: int = 2,
        stride: int = 1,
        kernel_size: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        encoder_type: str = "conv",
        pe_type: str = "neuron",
        pe_mode: str = "concat",
        num_pe_neuron: int = 40,
        neuron_pe_scale: float = 1000.0,
    ):
        super().__init__()
        
        self.hidden_size   = args.hidden_size  
        self.num_steps = args.T    
        self.input_size = args.feature_size  
        self.feature_size = args.feature_size
        self.pre_length = args.pre_length      
        self.num_levels = args.blocks
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.kernel_size = args.kernel_size
        self.args = args
        


        self._snn_backend = "snntorch"
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](self.hidden_size)
        

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
                SpikeTemporalBlock(
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
            self.__output_size = self.feature_size + num_pe_neuron
        else:
            self.__output_size = args.seq_length

        self.fc1 = nn.Linear(self.__output_size, args.feature_size)
        self.fc2 = nn.Linear(args.seq_length, self.pre_length)
        self.to('cuda:0') 
    
    def forward(self, inputs: torch.Tensor):
        utils.reset(self.encoder)

        if self.args.normalize:

            mean = inputs.mean(dim=1, keepdim=True).detach() # shape [B, 1, D]
            inputs = inputs - mean

            std = torch.sqrt(torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            inputs = inputs / std

        inputs = self.encoder(inputs)  # B, H, C, L
        # inputs: 24, 64, 321, 168
        
    
        if self.pe_type != "none":
            inputs = self.pe(inputs.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
        
        spks = self.network(inputs)
        spks = spks.squeeze(1)  # B, C', L
        preds = self.fc1(spks.permute(0, 2, 1))  # B, L, C
        preds = self.fc2(preds.permute(0, 2, 1))  # B, C', L
        preds = preds.permute(0, 2, 1).contiguous()
        if self.args.normalize:
            preds = preds * std + mean  # denormalize
        
        
        # Create auxiliary output
        aux = {'gate_l0': torch.tensor(0.0, device=preds.device)} # placeholder
        
        return preds, aux
    
    @property
    def output_size(self):
        return self.__output_size
