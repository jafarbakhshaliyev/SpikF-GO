from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based import surrogate



class CPGSpikePE(nn.Module):
    """
    Spike-form positional encoding (CPG-PE).
    Generates 2*N_pe binary channels with log-spaced rhythms over the flattened index t in [0, T*M).
    Shapes:
      returns pe: [T, B, M, 2*N_pe] with 0/1 spikes (no learnable params).
    """
    def __init__(self,
                 num_pairs: int = 20,
                 tau: float = 10000.0,
                 eta: float = 1.0,
                 vthres: float = 0.8,
                 w_max: float = 10000.0):
        super().__init__()
        self.num_pairs = num_pairs
        self.tau = tau
        self.eta = eta
        self.vthres = vthres
        self.w_max = w_max

    def forward(self, T: int, B: int, M: int, device) -> torch.Tensor:
        # t index across T*M (as recommended, flatten T and position to ensure uniqueness per step)
        t = torch.arange(T * M, device=device, dtype=torch.float32)  # [T*M]

        # log-spaced “frequencies” (same idea as CPGLinear): exp(-log(w_max) * i / N)
        # we only need relative spacing; this mirrors the paper’s log scaling.
        i = torch.arange(self.num_pairs, device=device, dtype=torch.float32)
        freq = torch.exp(-torch.log(torch.tensor(self.w_max, device=device)) * (i / max(1, self.num_pairs)))  # [N_pe]

        # phase arguments
        arg = self.eta * (t[:, None] * freq[None, :] / self.tau)  # [T*M, N_pe]

        # Heaviside on sin/cos with high threshold (~0.8) → sparse binary spikes
        cos_spk = (torch.cos(arg) - self.vthres > 0).float()
        sin_spk = (torch.sin(arg) - self.vthres > 0).float()

        pe = torch.cat([cos_spk, sin_spk], dim=1)                      # [T*M, 2*N_pe]
        pe = pe.view(T, M, 2 * self.num_pairs).unsqueeze(1)            # [T, 1, M, 2*N_pe]
        pe = pe.expand(-1, B, -1, -1).contiguous()                     # [T, B, M, 2*N_pe]
        return pe

# ---------------------------------------------------------
# K-bit quantizer with Straight-Through Estimator (STE)
# ---------------------------------------------------------
class KBitSteQuantizer(nn.Module):
    """
    K-bit symmetric quantizer with STE.
    - Forward: quantizes x to k-bit integers (symmetric around 0) and dequantizes.
    - Backward: gradient is approx. identity (straight-through).
    This is applied AFTER tok * s_bin, so zeros remain zeros (no spike, no event).
    """
    def __init__(self, k_bits: int = 4, eps: float = 1e-8):
        super().__init__()
        assert k_bits >= 1, "k_bits must be >= 1"
        self.k_bits = k_bits
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: arbitrary real tensor
        Returns: quantized-dequantized tensor with STE gradient.
        """
        # If bitwidth somehow disabled, fall back to identity
        if self.k_bits is None:
            return x

        with torch.no_grad():
            x_detached = x.detach()
            # Dynamic range per-tensor (can also make per-channel if you want)
            alpha = x_detached.abs().max()
            if alpha < self.eps:
                # All zeros or extremely small -> nothing to quantize
                x_q = x_detached
            else:
                qmax = 2 ** (self.k_bits - 1) - 1  # e.g. 7 for 4-bit ([-7,7])
                # Scale to [-qmax, qmax]
                x_scaled = x_detached / alpha * qmax
                x_rounded = torch.round(x_scaled)
                x_clamped = torch.clamp(x_rounded, -qmax, qmax)
                # Dequantize back to real domain
                x_q = x_clamped / qmax * alpha

        # STE: forward uses x_q, backward uses gradient of identity on x
        return x + (x_q - x_detached)


# ---------------------------------------------------------
# SFFT with k-bit quantization on tok * s_bin
# ---------------------------------------------------------
class SFFT(nn.Module):
    """
    Vectorized cuFFT version with optional k-bit quantization:
      - rfft_masked(tok, s_bin):  amplitude-aware (spike-masked) FFT
        -> optionally quantized to k-bit integers via KBitSteQuantizer
      - irfft_batched(Z_t):       batched inverse
    Shapes:
      tok:   [B, M, E] (real)
      s_bin: [T, B, M, E] (0/1 float or bool)
      Z_t:   [T, B, F, E] (complex)
    """
    def __init__(self, M: int, k_bits: Optional[int] = None):
        super().__init__()
        self.M = M
        self.F = M // 2 + 1

        # Optional k-bit quantizer on tok * s_bin
        self.k_bits = k_bits
        self.quantizer = KBitSteQuantizer(k_bits=k_bits) if k_bits is not None else None

    def rfft_masked(self, tok: torch.Tensor, s_bin: torch.Tensor) -> torch.Tensor:
        """
        tok:   [B, M, E] (float)
        s_bin: [T, B, M, E] (0/1)
        Returns complex spectrum Z_t: [T, B, F, E]
        """
        T, B, M, E = s_bin.shape
        assert tok.shape == (B, M, E), f"tok {tok.shape} vs (B,M,E) {(B,M,E)}"

        # Multiply by binary spikes in time
        x = tok.unsqueeze(0) * s_bin  # [T,B,M,E]; zeros where no spike

        # K-bit quantization (simulated integer, STE for backprop)
        if self.quantizer is not None:
            x = self.quantizer(x)  # still [T,B,M,E]

        # Flatten for batched rFFT over length M
        x = x.permute(0, 1, 3, 2).contiguous().view(T * B * E, M)  # [T*B*E, M]
        Z = torch.fft.rfft(x, n=self.M, dim=-1, norm='ortho')       # [T*B*E, F]
        Z = Z.view(T, B, E, self.F).permute(0, 1, 3, 2).contiguous()  # [T,B,F,E]
        return Z

    def irfft_batched(self, Z_t: torch.Tensor) -> torch.Tensor:
        T, B, F_bins, E = Z_t.shape
        x = Z_t.permute(0, 1, 3, 2).contiguous().view(T * B * E, F_bins)  # [T*B*E, F]
        x = torch.fft.irfft(x, n=self.M, dim=-1, norm='ortho')            # [T*B*E, M]
        x = x.view(T, B, E, self.M).permute(0, 1, 3, 2).contiguous()      # [T,B,M,E]
        return x


# ---------------------------------------------------------
# HardConcrete frequency gate (unchanged)
# ---------------------------------------------------------
class HardConcreteGate(nn.Module):
    """
    Hard-Concrete gate over frequency bins (shared across T and channels).
    Produces mask m in [0,1]; at train: stochastic; at eval: deterministic mean.
    Input/Output: complex tensor [T, B, F, E]; returns (masked_Z, mask_m).
    """
    def __init__(self, F_bins: int, init_logit: float = 2.0, beta: float = 2./3., eps: float = 1e-6):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((F_bins,), init_logit))
        self.beta = beta
        self.eps = eps

    def _sample_u(self, shape, device):
        return torch.empty(shape, device=device).uniform_(self.eps, 1 - self.eps)

    def _hard_concrete(self, training: bool, device, tau: float = 0.1):
        if training:
            u = self._sample_u(self.log_alpha.shape, device)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / tau)
        else:
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * 1.2 - 0.1
        return s_bar.clamp(0, 1)

    def forward(self, Z: torch.Tensor, tau: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        # Z: [T, B, F, E] complex
        m = self._hard_concrete(self.training, Z.device, tau)  # [F]
        m = m.view(1, 1, -1, 1).to(Z.real)
        return Z * m, m  # masked Z and mask

    def l0(self) -> torch.Tensor:
        p = torch.sigmoid(self.log_alpha)
        return p.mean()


# ---------------------------------------------------------
# Complex normalization and filters (unchanged)
# ---------------------------------------------------------
class ComplexLayerNormFreqChan(nn.Module):
    """LN on real part over channel dim (D), per (T,B,F,*)."""
    def __init__(self, D: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(D, eps=eps)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr = self.ln(z.real)
        return torch.complex(zr, z.imag)


class EventDrivenDepthwiseComplex(nn.Module):
    """
    Depthwise complex filter H(f,d) applied to z (per-frequency, per-channel).
    If a gate g_t is provided, multiplies only where g_t==1 (event-driven).
    Shapes: z, out: complex [T,B,F,D]; g_t: float/bool [T,B,F,D] or None.
    """
    def __init__(self, F_bins: int, D: int, hidden_size_factor, init_scale: float = 0.02):
        super().__init__()
        self.F, self.D, self.hidden_size_factor = F_bins, D, hidden_size_factor
        # 2 x [D,D] params for real/imag mixing
        self.w1 = nn.Parameter(init_scale * torch.randn(2, self.D, self.D * self.hidden_size_factor))
        self.b1 = nn.Parameter(init_scale * torch.randn(2, self.D * self.hidden_size_factor))

    def forward(self, z: torch.Tensor, g_t: torch.Tensor | None = None) -> torch.Tensor:
        T, B, F_bin, D = z.shape

        yr = torch.einsum('tbli,ij->tblj', z.real, self.w1[0]) - \
             torch.einsum('tbli,ij->tblj', z.imag, self.w1[1]) + \
             self.b1[0]

        yi = torch.einsum('tbli,ij->tblj', z.imag, self.w1[0]) + \
             torch.einsum('tbli,ij->tblj', z.real, self.w1[1]) + \
             self.b1[1]

        out = torch.complex(yr, yi)
        if g_t is not None:
            out = out * g_t
        return out


class ComplexLIFGate(nn.Module):
    """
    Per-(f,d) spiking decision using real & imaginary components separately.
    We compute s_r = LIF(x.real), s_i = LIF(x.imag), then a binary gate:
      g = 1 if (s_r > 0) or (s_i > 0), else 0
    This keeps sensitivity to BOTH real and imaginary parts; no |X|.
    """
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0):
        super().__init__()
        self.lif_r = MultiStepLIFNode(
            tau=tau, v_threshold=v_threshold, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
        )
        self.lif_i = MultiStepLIFNode(
            tau=tau, v_threshold=v_threshold, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s_r = self.lif_r(z.real)  # [T,B,F,D] in [0,1]
        s_i = self.lif_i(z.imag)
        g = ((s_r > 0) | (s_i > 0)).to(z.real.dtype)  # OR keeps phase sensitivity
        return g


class SpikingFGOConvKeepT_Cplx(nn.Module):
    """
    Convolution-theorem-faithful FGO block (3 stages), keeps T, preserves complex math.
    - Optional event-driven gating (component-wise) for energy; turn off to match original FGO.
    Shapes: in/out complex [T,B,F,D]
    """
    def __init__(self, F_bins: int, D: int, hidden_size_factor: int = 1,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 soft_shrink: float = 0.01, use_gate: bool = True):
        super().__init__()
        self.F, self.D = F_bins, D
        self.shr = soft_shrink
        self.use_gate = use_gate

        self.filt1 = EventDrivenDepthwiseComplex(F_bins, D, hidden_size_factor=1)
        self.filt2 = EventDrivenDepthwiseComplex(F_bins, D, hidden_size_factor=1)
        self.filt3 = EventDrivenDepthwiseComplex(F_bins, D, hidden_size_factor=1)

        self.gate1 = ComplexLIFGate(tau=tau, v_threshold=v_threshold)
        self.gate2 = ComplexLIFGate(tau=tau, v_threshold=v_threshold)
        self.gate3 = ComplexLIFGate(tau=tau, v_threshold=v_threshold)

        self.ln1 = ComplexLayerNormFreqChan(D)
        self.ln2 = ComplexLayerNormFreqChan(D)
        self.ln3 = ComplexLayerNormFreqChan(D)

    @staticmethod
    def _shrink(z: torch.Tensor, lambd: float) -> torch.Tensor:
        return torch.complex(F.softshrink(z.real, lambd), F.softshrink(z.imag, lambd))

    def forward(self, X: torch.Tensor):
        stats = {}

        # --- Block 1 ---
        G1 = self.gate1(X) if self.use_gate else None
        Y  = self.filt1(self.ln1(X), G1)
        Y  = self._shrink(Y, self.shr)
        if G1 is not None: stats['gate_frac_1'] = G1.mean()

        # --- Block 2 + residual ---
        G2 = self.gate2(Y) if self.use_gate else None
        Z  = self.filt2(self.ln2(Y), G2)
        Z  = self._shrink(Z, self.shr)
        X  = X + Z
        if G2 is not None: stats['gate_frac_2'] = G2.mean()

        # --- Block 3 + residual ---
        G3 = self.gate3(X) if self.use_gate else None
        W  = self.filt3(self.ln3(X), G3)
        W  = self._shrink(W, self.shr)
        Out = X + W
        if G3 is not None: stats['gate_frac_3'] = G3.mean()

        return Out, stats


# ---------------------------------------------------------
# T-preserving spiking decoder (unchanged)
# ---------------------------------------------------------
class SpikingDecoderKeepT(nn.Module):
    """
    Keeps T through spiking blocks; collapses at the very end with learned softmax over T.
    Input  y_t: [T,B,N,E,L]
    Output pred: [B, pred_len, N]
    """
    def __init__(self, D: int, L: int, pred_len: int,
                 hidden: int = 256, proj_dim: int = 8,
                 T: int = 4, tau: float = 2.0, v_th: float = 1.0, use_ln: bool = True):
        super().__init__()
        self.T = T
        self.pred_len = pred_len
        self.use_ln = use_ln

        self.time_basis = nn.Parameter(torch.randn(L, proj_dim) * 0.02)
        self.time_proj = nn.Sequential(
            nn.Linear(L, proj_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim, bias=False)
        )

        in_feat = D * proj_dim
        self.lin1 = nn.Linear(in_feat, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, hidden)
        self.skip = nn.Linear(in_feat, hidden, bias=False)

        if use_ln:
            self.ln1 = nn.LayerNorm(512)
            self.ln2 = nn.LayerNorm(256)
            self.ln3 = nn.LayerNorm(hidden)

        self.lif1 = MultiStepLIFNode(
            tau=tau, v_threshold=v_th, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
        )
        self.lif2 = MultiStepLIFNode(
            tau=tau*1.5, v_threshold=v_th*0.8, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
        )
        self.lif3 = MultiStepLIFNode(
            tau=tau*0.7, v_threshold=v_th*1.2, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
        )

        self.lin_out = nn.Linear(hidden, pred_len)
        self.alpha_T = nn.Parameter(torch.zeros(T))  # learned readout weights over T

        # init
        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.lin_out.weight, gain=0.2)
        nn.init.zeros_(self.lin_out.bias)
        nn.init.orthogonal_(self.skip.weight, gain=0.1)

    def forward(self, y_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        y_t: [T,B,N,E,L]
        ->   [B, pred_len, N]
        """
        T, B, N, E, L = y_t.shape
        assert T == self.T

        # Project along L
        y_proj = self.time_proj(y_t) + torch.matmul(y_t, self.time_basis)  # [T,B,N,E,P]
        y_flat = y_proj.reshape(T, B * N, -1)  # [T, B*N, E*P]

        skip = self.skip(y_flat)  # [T, B*N, hidden]

        h = self.lin1(y_flat)
        if self.use_ln: h = self.ln1(h)
        h = self.lif1(h)

        h = self.lin2(h)
        if self.use_ln: h = self.ln2(h)
        h = self.lif2(h)

        h = self.lin3(h)
        if self.use_ln: h = self.ln3(h)
        h = self.lif3(h + 0.1 * skip)  # [T, B*N, hidden]

        logits_t = self.lin_out(h)  # [T, B*N, pred_len]
        w = torch.softmax(self.alpha_T, dim=0).view(T, 1, 1)
        logits = (w * logits_t).sum(dim=0)  # [B*N, pred_len]

        preds = logits.view(B, N, self.pred_len).permute(0, 2, 1).contiguous()
        stats = {'readout_T_entropy': -(w.squeeze() * torch.log(w.squeeze() + 1e-8)).sum()}
        return preds, stats


# ---------------------------------------------------------
# Main SpikF_GO model with k-bit SFFT
# ---------------------------------------------------------
class SpikF_GO1_CPG(nn.Module):
    """
    Pipeline:
      x [B,L,N] -> flatten M=N*L -> embed -> spike-encode over T -> S-FFT (event-driven, k-bit quantized)
      -> HardConcrete gating -> Spiking Fourier block (event-driven diagonal scaling)
      -> S-iFFT (event-driven) -> reshape -> spiking decoder (keeps T)
      -> learned T readout -> preds [B,P,N]
    """
    def __init__(self, args, pre_length, embed_size,
                 feature_size, seq_length, hidden_size,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1,
                 sparsity_threshold=0.01,
                 k_bits: int = 8):
        super().__init__()
        self.args = args
        E, N, L = embed_size, feature_size, seq_length
        self.M = N * L
        self.F = self.M // 2 + 1
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor

        self.use_cpg_pe = getattr(args, 'use_cpg_pe', True)
        self.num_pe_pairs = getattr(args, 'num_pe_pairs', 20)
        self.pe_tau = getattr(args, 'pe_tau', 10000.0)
        self.pe_eta = getattr(args, 'pe_eta', 1.0)
        self.pe_vthres = getattr(args, 'pe_vthres', 0.8)
        self.pe_wmax = getattr(args, 'pe_wmax', 10000.0)

        if self.use_cpg_pe:
            self.cpg_pe = CPGSpikePE(
                num_pairs=self.num_pe_pairs,
                tau=self.pe_tau, eta=self.pe_eta,
                vthres=self.pe_vthres, w_max=self.pe_wmax
            )
            # Map [E + 2*N_pe] → E, then BN + LIF to keep spikes (Eq. (13)–(15) idea)
            self.pe_linear = nn.Linear(E + 2 * self.num_pe_pairs, E, bias=False)
            self.pe_bn = nn.BatchNorm1d(E)
            self.pe_lif = MultiStepLIFNode(
                tau=self.args.tau, v_threshold=self.args.alpha, detach_reset=True,
                surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
            )

        # Token embedding: scalar -> E
        self.embeddings = nn.Parameter(torch.randn(1, E) * 0.02)
        self.enc_ln = nn.LayerNorm(self.M)
        self.enc_gain = nn.Parameter(torch.ones(1))

        # Step modulation to make steps non-identical
        self.step_gamma = nn.Parameter(torch.ones(args.T))
        self.step_beta = nn.Parameter(torch.zeros(args.T))
        self.register_buffer('step_scale', torch.linspace(0, 1, steps=args.T).view(args.T, 1, 1, 1))

        # Encoder LIF for spike generation
        self.encoder_lif = MultiStepLIFNode(
            tau=args.tau, v_threshold=args.alpha, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=4.0), backend='torch'
        )

        # SFFT with k-bit quantization on tok * s_bin
        self.sfft = SFFT(self.M, k_bits=k_bits)

        # Hard-Concrete frequency pruning
        self.freq_gate = HardConcreteGate(self.F, init_logit=2.0)
        self.register_buffer('gate_tau', torch.tensor(0.10))  # annealed in training loop

        # Event-driven Fourier block (keeps T)
        self.sf_block = SpikingFGOConvKeepT_Cplx(
            F_bins=self.F, D=args.embed_size, hidden_size_factor=self.hidden_size_factor,
            tau=args.tau, v_threshold=args.alpha,
            soft_shrink=self.sparsity_threshold,
            use_gate=True   # set False to exactly match FGO (no gating)
        )

        # T-preserving decoder
        self.decoder = SpikingDecoderKeepT(
            D=E, L=L, pred_len=args.pre_length, hidden=args.hidden_size,
            proj_dim=args.proj_dim, T=args.T, tau=args.tau, v_th=args.alpha, use_ln=True
        )

    def token_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N] -> flatten [B, M] and expand to [B,M,E]
        B, L, N = x.shape
        x_flat = x.permute(0, 2, 1).contiguous().reshape(B, self.M)  # [B,M]
        return x_flat.unsqueeze(-1) * self.embeddings  # [B,M,E]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, L, N]
        -> preds: [B, P, N]
           aux:   dict of monitors (gate L0, gating fractions, etc.)
        """
        B, L, N = x.shape
        E, T = self.args.embed_size, self.args.T

        # --------- Normalize input (per series) ----------
        mean = x.mean(dim=1, keepdim=True).detach()  # [B,1,N]
        x = x - mean

        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        # --------- 1) Embed tokens ----------
        tok = self.token_embed(x)    # [B,M,E], real
        tok = tok.transpose(1, 2)    # [B,E,M]
        tok = self.enc_ln(tok)       # LN over M
        tok = tok.transpose(1, 2).contiguous()  # [B,M,E]
        tok = tok * self.enc_gain

        # --------- 2) Build T-step spike events for S-FFT ----------
        cur_t = tok.unsqueeze(0).repeat(T, 1, 1, 1)  # [T,B,M,E]
        cur_t = cur_t * self.step_gamma.view(T, 1, 1, 1) + self.step_beta.view(T, 1, 1, 1)
        cur_t = cur_t * (1 + 0.02 * self.step_scale.to(cur_t.dtype))
        s_t = self.encoder_lif(cur_t)   # [T,B,M,E] in [0,1]
        s_bin = (s_t > 0).float()       # event mask (0/1)
        enc_rate_cont = s_t.mean()      # differentiable
        enc_rate_bin  = s_bin.mean().detach()
        
        if self.use_cpg_pe:
            # build spike-form PE and concatenate along channel dim
            pe_spk = self.cpg_pe(T=T, B=B, M=self.M, device=x.device)      # [T,B,M,2*N_pe]
            s_cat = torch.cat([s_t, pe_spk], dim=-1)                        # [T,B,M,E+2*N_pe]

            # Linear → BN → LIF (multi-step) to return to spike mask with size E
            h = self.pe_linear(s_cat)                                       # [T,B,M,E]
            h = h.reshape(T * B * self.M, E)
            h = self.pe_bn(h).view(T, B, self.M, E)
            s_t = self.pe_lif(h)  

        # --------- 3) K-bit masked S-FFT ----------
        Z_t = self.sfft.rfft_masked(tok, s_bin)  # [T,B,F,E] complex

        # --------- 4) HardConcrete frequency gate ----------
        Z_t, m = self.freq_gate(Z_t, tau=float(self.gate_tau))

        # --------- 5) Spiking Fourier block (complex) ----------
        Z_t, sf_stats = self.sf_block(Z_t)

        rho_mean = torch.real(m).mean().detach()

        # --------- 6) Inverse S-FFT back to time  ----------
        y_time_t = self.sfft.irfft_batched(Z_t)  # [T,B,M,E]
        # reshape [T,B,N,L,E] -> [T,B,N,E,L]
        y_t = y_time_t.view(T, B, N, L, E).permute(0, 1, 2, 4, 3).contiguous()  # [T,B,N,E,L]

        # --------- 7) T-preserving decoder + learned T readout ----------
        preds, dec_stats = self.decoder(y_t)         # [B,P,N]
        preds = preds * std + mean                   # de-normalize

        aux = {
            'enc_rate': enc_rate_cont,                  # use this for spike-rate reg if wanted
            'enc_rate_bin': enc_rate_bin,
            'rho_mean': rho_mean,
            'rho_hat': self.freq_gate.l0(),
            'freq_mask_active': (torch.abs(m) > 0.5).cpu().float().mean().detach(),
            **sf_stats,
            **dec_stats
        }
        return preds.permute(0, 2, 1), aux
