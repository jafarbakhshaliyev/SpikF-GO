from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based import surrogate



class TokenRMSNormOverM(nn.Module):
    """
    tok: [B, M, E]
    Normalize over M (token axis) per sample, per channel.
    Hardware-friendly vs LN: no mean subtraction, only rms.
    """
    def __init__(self, E: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(E))
            self.beta  = nn.Parameter(torch.zeros(E))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        # rms over M
        rms = torch.rsqrt(tok.pow(2).mean(dim=1, keepdim=True) + self.eps)  # [B,1,E]
        y = tok * rms
        if self.affine:
            y = y * self.gamma + self.beta
        return y



class SFFT(nn.Module):
    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.F = M // 2 + 1

    def rfft(self, s_t: torch.Tensor) -> torch.Tensor:
        # s_t: [T, B, M, E] real
        T, B, M, E = s_t.shape
        x = s_t.permute(0, 1, 3, 2).contiguous().view(T * B * E, M)  # [T*B*E, M]
        Z = torch.fft.rfft(x, n=self.M, dim=-1, norm="ortho")        # [T*B*E, F] complex
        Z = Z.view(T, B, E, self.F).permute(0, 1, 3, 2).contiguous() # [T,B,F,E]
        return Z

    def irfft(self, Z_t: torch.Tensor) -> torch.Tensor:
        # Z_t: [T, B, F, E] complex
        T, B, Freq, E = Z_t.shape
        x = Z_t.permute(0, 1, 3, 2).contiguous().view(T * B * E, Freq)  # [T*B*E, F]
        y = torch.fft.irfft(x, n=self.M, dim=-1, norm="ortho")          # [T*B*E, M]
        y = y.view(T, B, E, self.M).permute(0, 1, 3, 2).contiguous()    # [T,B,M,E]
        return y



class HardConcreteGate(nn.Module):
    """
    Gate over frequency bins, shared across T,B,E.
    Z: [T,B,F,E] complex
    mask m: [1,1,F,1] in [0,1]
    """
    def __init__(self, F_bins: int, init_logit: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((F_bins,), float(init_logit)))
        self.eps = eps

    def _sample_u(self, shape, device):
        return torch.empty(shape, device=device).uniform_(self.eps, 1.0 - self.eps)

    def _hard_concrete(self, training: bool, device, tau: float):
        if training:
            u = self._sample_u(self.log_alpha.shape, device)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / tau)
        else:
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * 1.2 - 0.1
        return s_bar.clamp(0.0, 1.0)

    def forward(self, Z: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self._hard_concrete(self.training, Z.device, tau=tau)  # [F]
        m = m.view(1, 1, -1, 1).to(Z.real.dtype)                   # [1,1,F,1]
        return Z * m, m

    def l0(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha).mean()



class Affine(nn.Module):
    """Per-channel affine: y = x * gamma + beta for real tensors [..., D]."""
    def __init__(self, D: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(D))
        self.beta  = nn.Parameter(torch.zeros(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta


class ComplexAffine(nn.Module):
    """Per-channel affine for complex tensors on last dim E."""
    def __init__(self, E: int):
        super().__init__()
        self.gamma_r = nn.Parameter(torch.ones(E))
        self.beta_r  = nn.Parameter(torch.zeros(E))
        self.gamma_i = nn.Parameter(torch.ones(E))
        self.beta_i  = nn.Parameter(torch.zeros(E))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr = z.real * self.gamma_r + self.beta_r
        zi = z.imag * self.gamma_i + self.beta_i
        return torch.complex(zr, zi)



class ComplexLinear(nn.Module):
    """
    x: [T,B,F,E_in] complex -> [T,B,F,E_out] complex
    """
    def __init__(self, E_in: int, E_out: int, init_scale: float = 0.02):
        super().__init__()
        self.Wr = nn.Parameter(init_scale * torch.randn(E_in, E_out))
        self.Wi = nn.Parameter(init_scale * torch.randn(E_in, E_out))
        self.br = nn.Parameter(torch.zeros(E_out))
        self.bi = nn.Parameter(torch.zeros(E_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.real, x.imag
        yr = xr @ self.Wr - xi @ self.Wi + self.br
        yi = xi @ self.Wr + xr @ self.Wi + self.bi
        return torch.complex(yr, yi)


class ComplexLIFGate(nn.Module):
    """
    Turns complex activations into a binary gate g in {0,1} (float),
    based on spikes from real or imag parts.
      g = 1 if spike(real) OR spike(imag)
    """
    def __init__(self, tau: float, v_th: float):
        super().__init__()
        self.lif_r = MultiStepLIFNode(
            tau=tau, v_threshold=v_th, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend="torch"
        )
        self.lif_i = MultiStepLIFNode(
            tau=tau, v_threshold=v_th, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend="torch"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [T,B,F,D] complex
        s_r = self.lif_r(z.real)  # [T,B,F,D] in [0,1]
        s_i = self.lif_i(z.imag)
        g = ((s_r > 0) | (s_i > 0)).to(z.real.dtype)
        return g



class SpikingFourierBlock(nn.Module):
    """
    3-stage residual freq block:

      Stage1:  A1 = Affine(Z)  -> GA1 -> gate -> Lin1 -> G1 -> gate
      Stage2:  A2 = Affine(Y)  -> GA2 -> gate -> Lin2 -> G2 -> gate -> ReZero residual
      Stage3:  A3 = Affine(Z)  -> GA3 -> gate -> Lin3 -> G3 -> gate -> ReZero residual

    This keeps the signal "spike-like" before the heavy mixing.
    """
    def __init__(
        self,
        args,
        E: int,
        hidden_size_factor: int,
        tau: float = 2.0,
        v_th: float = 1.0,
        apply_gate_to_complex: bool = True,
    ):
        super().__init__()
        H = int(E * hidden_size_factor)

        self.args = args

        self.lin1 = ComplexLinear(E, H)
        self.lin2 = ComplexLinear(H, E)
        self.lin3 = ComplexLinear(E, E)


        # gates after linears (as before)
        self.g1 = ComplexLIFGate(tau=tau, v_th=v_th)   # after lin1 (H)
        self.g2 = ComplexLIFGate(tau=tau, v_th=v_th)   # after lin2 (E)
        self.g3 = ComplexLIFGate(tau=tau, v_th=v_th)   # after lin3 (E)

        self.apply_gate_to_complex = apply_gate_to_complex

        # ReZero residual scaling
        self.r2 = nn.Parameter(torch.tensor(0.1))
        self.r3 = nn.Parameter(torch.tensor(0.1))

        if self.args.affine:
            self.a1 = ComplexAffine(E)
            self.a2 = ComplexAffine(H)
            self.a3 = ComplexAffine(E)
            self.ga1 = ComplexLIFGate(tau=tau, v_th=v_th)  
            self.ga2 = ComplexLIFGate(tau=tau, v_th=v_th)  
            self.ga3 = ComplexLIFGate(tau=tau, v_th=v_th)  


    def _apply_gate(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if not self.apply_gate_to_complex:
            return z
        return z * g.to(z.real.dtype)

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats: Dict[str, torch.Tensor] = {}

        # ---- Stage 1 ----

        if self.args.affine:        
            A1 = self.a1(Z)
            GA1 = self.ga1(A1)
            A1 = self._apply_gate(A1, GA1)
        else:
            A1 = Z

        Y = self.lin1(A1)
        G1 = self.g1(Y)
        Y = self._apply_gate(Y, G1)

        # ---- Stage 2 + residual ----
        if self.args.affine:
            A2 = self.a2(Y)
            GA2 = self.ga2(A2)
            A2 = self._apply_gate(A2, GA2)
        else:
            A2 = Y

        X = self.lin2(A2)
        G2 = self.g2(X)
        X = self._apply_gate(X, G2)

        Z2 = Z + self.r2 * X

        # ---- Stage 3 + residual ----

        if self.args.affine:
            A3 = self.a3(Z2)
            GA3 = self.ga3(A3)
            A3 = self._apply_gate(A3, GA3)
        else:
            A3 = Z2


        W = self.lin3(A3)
        G3 = self.g3(W)
        W = self._apply_gate(W, G3)

        out = Z2 + self.r3 * W

        with torch.no_grad():
            mag2 = out.real * out.real + out.imag * out.imag
            stats["freq_active_frac"] = (mag2 > 0).float().mean()

            stats["rezero_r2"] = self.r2.detach()
            stats["rezero_r3"] = self.r3.detach()

            stats["gate_lin_frac_1"] = G1.mean().detach()
            stats["gate_lin_frac_2"] = G2.mean().detach()
            stats["gate_lin_frac_3"] = G3.mean().detach()

        return out, stats



class Decoder(nn.Module):
    """
    y_t: [T,B,N,E,L] -> preds: [B,P,N]
    """
    def __init__(
        self,
        E: int,
        L: int,
        pred_len: int,
        T: int,
        tau: float,
        v_th: float,
        proj_dim: int = 4,
        reduced_dim: int = 64,
    ):
        super().__init__()
        self.E, self.L, self.P, self.T = E, L, pred_len, T
        self.proj_dim = int(proj_dim)

        self.time_proj = nn.Linear(L, self.proj_dim, bias=False)
        D_in = E * self.proj_dim
        self.reduced_dim = int(reduced_dim)

        self.lif = MultiStepLIFNode(
            tau=tau,
            v_threshold=v_th,
            detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0),
            backend="torch",
        )

        self.fc_reduce = weight_norm(nn.Linear(D_in, int(reduced_dim), bias=True))
        self.fc_out    = weight_norm(nn.Linear(int(reduced_dim), pred_len, bias=True))

        nn.init.xavier_uniform_(self.time_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc_reduce.weight, gain=0.6)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.2)
        nn.init.zeros_(self.fc_reduce.bias)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, y_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        T, B, N, E, L = y_t.shape

        y_p = self.time_proj(y_t)                        # [T,B,N,E,p]
        x   = y_p.reshape(T, B * N, E * self.proj_dim)   # [T,B*N,D]
        s   = self.lif(x)                                # [T,B*N,D] spikes
        h_t = self.fc_reduce(s.reshape(T * B * N, -1)).view(T, B * N, self.reduced_dim)

        h = h_t.mean(dim=0)                           # [B*N,reduced_dim]
        h = F.gelu(h)
        out = self.fc_out(h)                             # [B*N,P]

        preds = out.view(B, N, self.P).permute(0, 2, 1).contiguous()
        stats = {"dec_spike_rate": s.mean().detach()}
        return preds, stats



class SpikF_GO2(nn.Module):
    def __init__(
        self,
        args,
        pre_length: int,
        embed_size: int,
        feature_size: int,
        seq_length: int,
        hidden_size: int,
        hard_thresholding_fraction=1,
        hidden_size_factor: int = 1,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        self.args = args

        self.N = feature_size
        self.L = seq_length
        self.E = embed_size
        self.T = args.T
        self.M = self.N * self.L

        # token embedding (scalar -> E)
        self.embeddings = nn.Parameter(torch.randn(1, self.E) * 0.02)
        self.tok_aff = Affine(self.E)
        self.tok_rms = TokenRMSNormOverM(E=self.E, eps=1e-6, affine=True)

        #self.enc_gain = nn.Parameter(torch.tensor(1.0))

        # step modulation
        self.step_gamma = nn.Parameter(torch.ones(self.T))
        self.step_beta  = nn.Parameter(torch.zeros(self.T))
        self.register_buffer("step_scale", torch.linspace(0, 1, steps=self.T).view(self.T, 1, 1, 1))

        # encoder lif
        self.encoder_lif = MultiStepLIFNode(
            tau=args.tau,
            v_threshold=args.alpha,
            detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0),
            backend="torch",
        )

        # FFT
        self.sfft = SFFT(self.M)
        self.F_bins = self.sfft.F

        # frequency pruning
        self.freq_gate = HardConcreteGate(self.F_bins, init_logit=2.0)
        self.register_buffer("gate_tau", torch.tensor(0.10))


        self.freq_block = SpikingFourierBlock(
            self.args,
            E=self.E,
            hidden_size_factor=hidden_size_factor,
            tau=args.tau,
            v_th=args.alpha,
            apply_gate_to_complex=True,
        )

        # decoder
        proj_dim = self.args.proj_dim
        reduced_dim = max(16, min(128, hidden_size // 4))
        self.decoder = Decoder(
            E=self.E,
            L=self.L,
            pred_len=pre_length,
            T=self.T,
            tau=args.tau,
            v_th=args.alpha,
            proj_dim=proj_dim,
            reduced_dim=reduced_dim,
        )

    def token_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,N] -> [B,M,E]
        B, L, N = x.shape
        x_flat = x.permute(0, 2, 1).contiguous().reshape(B, self.M)  # [B,M]
        tok = x_flat.unsqueeze(-1) * self.embeddings                 # [B,M,E]
        tok = self.tok_aff(tok)
        return tok
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, L, N = x.shape

        # normalize
        if self.args.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()
            x0 = x - mean
            std = torch.sqrt(torch.var(x0, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x0 = x0 / std
        else:
            mean, std = None, None
            x0 = x


        tok = self.token_embed(x0)         # [B,M,E]
        tok = self.tok_rms(tok)            # RMSNorm over M (hardware-friendly)


        # step modulation
        cur_t = tok.unsqueeze(0).repeat(self.T, 1, 1, 1)
        cur_t = cur_t * self.step_gamma.view(self.T, 1, 1, 1) + self.step_beta.view(self.T, 1, 1, 1)
        cur_t = cur_t * (1.0 + 0.02 * self.step_scale.to(cur_t.dtype))


        # spikes
        s_t = self.encoder_lif(cur_t)
        enc_rate = s_t.mean()

        # FFT
        Z_t = self.sfft.rfft(s_t)

        # prune
        Z_t, m = self.freq_gate(Z_t, tau=float(self.gate_tau))

        # freq block (+ extra LIF after affines)
        Z_t, fb_stats = self.freq_block(Z_t)

        # iFFT
        y_time_t = self.sfft.irfft(Z_t).to(tok.dtype)

        # reshape -> [T,B,N,E,L]
        y_t = y_time_t.view(self.T, B, N, self.L, self.E).permute(0, 1, 2, 4, 3).contiguous()

        preds, dec_stats = self.decoder(y_t)

        if self.args.normalize:
            preds = preds * std + mean

        aux = {
            "enc_rate": enc_rate.detach(),
            "rho_hat": self.freq_gate.l0().detach(),
            "freq_mask_mean": m.mean().detach(),
            "freq_mask_active": (m > 0.5).float().mean().detach(),
            **fb_stats,
            **dec_stats,
        }
        return preds, aux
