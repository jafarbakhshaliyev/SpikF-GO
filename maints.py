# main_ts_with_scaled_and_inverse_metrics.py
# - Computes & prints BOTH metrics:
#     (1) SCALED metrics (on scaled data)
#     (2) ORIG  metrics (after inverse_transform back to original scale)
# - Saves summary for BOTH (scaled + orig) across 5 runs
# - Uses train-set scaler for inverse_transform (no leakage)
# - Reuses same scaler for val/test by passing scaler=train_set.scaler (if your Dataset supports it)

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch as snn
import time
import os
import numpy as np
import warnings
from spikingjelly.clock_driven import functional

from data.data_loader import (
    Dataset_ECG, Dataset_Dhfm, Dataset_Solar, Dataset_Wiki, Dataset_PEMS_BAY, Dataset_PEMS
)
from utils.utils import save_model_ts, load_model_ts, evaluate

from model.SpikF_GO import SpikF_GO
from model.SpikF_GO1 import SpikF_GO1
from model.SpikF_GO2 import SpikF_GO2
from model.SpikF_GO1_CPG import SpikF_GO1_CPG
from model.SpikF_GO2_CPG import SpikF_GO2_CPG
from model.FourierGNN import FGN
from model.SpikF import SpikF
from model.iSpikformer import iSpikformer
from model.SpikF_GO_CPG import SpikF_GO_CPG
from model.TS_GRU import TSGRU
from model.TS_TCN import TSTCN
from model.TS_Former import TSFormer
from model.spikegru import SpikeGRU
from model.spikformer_cpg import SpikformerCPG
from model.spikernn import SpikeRNN
from model.spiketcn import SpikeTCN
from model.TS_TCN import TSLIFNode


# -----------------------------
# Spiking reset helpers
# -----------------------------
def remove(model):
    """Reset states of spiking neurons with warning suppression"""
    if model is None:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not base.MemoryModule.*")
        if hasattr(model, '__iter__'):
            for m in model:
                if hasattr(m, 'reset'):
                    m.reset()
                elif hasattr(m, 'v'):
                    m.v = 0.0
        elif hasattr(model, 'reset'):
            model.reset()
        elif hasattr(model, 'v'):
            model.v = 0.0


def reset_states(model):
    """Reset states of all spiking neurons (TSLIFNode, Leaky, etc.) with warning suppression."""
    if model is None:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not base.MemoryModule.*")
        if hasattr(model, '__iter__'):
            for m in model:
                reset_states(m)
        elif hasattr(model, 'modules'):
            for module in model.modules():
                if isinstance(module, (snn.Leaky, TSLIFNode)):
                    try:
                        module.reset()
                    except Exception:
                        if hasattr(module, 'v'):
                            module.v = 0.0
        elif hasattr(model, 'reset'):
            model.reset()
        elif hasattr(model, 'v'):
            model.v = 0.0


# -----------------------------
# Metrics helpers (scaled + inverse)
# -----------------------------
def _inverse_if_possible(arr: np.ndarray, scaler):
    """
    Inverse-transform arr of shape (..., D) using scaler fitted on train.
    If scaler is None, returns arr unchanged.
    """
    if scaler is None:
        return arr
    if not hasattr(scaler, "inverse_transform"):
        print("WHAT THE FUCK")
        return arr

    if arr.ndim < 2:
        return arr

    D = arr.shape[-1]
    flat = arr.reshape(-1, D)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(arr.shape)


def compute_scores_scaled_and_orig(trues: np.ndarray, preds: np.ndarray, scaler):
    """
    Returns:
      score_scaled = (mape, mae, rmse, r2) on scaled arrays
      score_orig   = (mape, mae, rmse, r2) on inverse-transformed arrays
    """
    score_scaled = evaluate(trues, preds)

    trues_inv = _inverse_if_possible(trues, scaler)
    preds_inv = _inverse_if_possible(preds, scaler)
    score_orig = evaluate(trues_inv, preds_inv)

    return score_scaled, score_orig


def _fmt_score(tag, score):
    mape, mae, rmse, r2, rse = score
    mape_pct = mape * 100.0
    return f"{tag}: MAPE {mape_pct:10.6f}; MAE {mae:10.6f}; RMSE {rmse:10.6f}; R2 {r2:10.6f}; RSE {rse:10.6f}."


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='ECG', help='data set')
parser.add_argument('--feature_size', type=int, default=140, help='feature size')
parser.add_argument('--seq_length', type=int, default=12, help='input length')
parser.add_argument('--pre_length', type=int, default=12, help='predict length')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=4, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--train_ratio', type=float, default=0.6)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--tau', type=float, default=2.0, help='tau')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--T', type=int, default=16)
parser.add_argument('--proj_dim', type=int, default=16, help='proj dim')
parser.add_argument('--model', type=str, default='FGN', help='model name')

parser.add_argument('--patch_num', type=int, default=48)
parser.add_argument('--patch_dim', type=int, default=32)
parser.add_argument('--blocks', type=int, default=1)
parser.add_argument('--energy_loss', type=bool, default=False)
parser.add_argument('--normalize', action='store_false', help='Disable normalization')
parser.add_argument('--affine', action='store_false', help='Disable affine layer')

# TS-TCN specific
parser.add_argument('--kernel_size', type=int, default=16)

args = parser.parse_args()
print(f'Training configs: {args}')


# -----------------------------
# Data config
# -----------------------------
data_parser = {
    'traffic':      {'root_path': 'data/traffic.npy',    'type': '0'},
    'ECG':          {'root_path': 'data/ECG_data.csv',   'type': '0'},
    'COVID':        {'root_path': 'data/covid.csv',      'type': '0'},
    'electricity':  {'root_path': 'data/electricity.csv','type': '0'},
    'solar':        {'root_path': './data/solar',        'type': '0'},
    'metr':         {'root_path': 'data/metr.csv',       'type': '0'},
    'wiki':         {'root_path': 'data/wiki.csv',       'type': '0'},
    'pems_bay':     {'root_path': 'data/pems-bay.h5',    'type': '0'},
    'pems03':       {'root_path': 'data/PEMS03.npz',     'type': '0'},
    'pems04':       {'root_path': 'data/PEMS04.npz',     'type': '0'},
    'pems07':       {'root_path': 'data/PEMS07.npz',     'type': '0'},
    'pems08':       {'root_path': 'data/PEMS08.npz',     'type': '0'},
}

data_dict = {
    'ECG':         Dataset_ECG,
    'COVID':       Dataset_ECG,
    'traffic':     Dataset_Dhfm,
    'solar':       Dataset_Solar,
    'wiki':        Dataset_Wiki,
    'electricity': Dataset_ECG,
    'metr':        Dataset_ECG,
    'pems_bay':    Dataset_PEMS_BAY,
    'pems03':      Dataset_PEMS,
    'pems04':      Dataset_PEMS,
    'pems07':      Dataset_PEMS,
    'pems08':      Dataset_PEMS,
}

if args.data not in data_parser:
    raise ValueError(f"Unknown dataset {args.data}. Available: {list(data_parser.keys())}")

data_info = data_parser[args.data]
Data = data_dict[args.data]


# -----------------------------
# Build datasets (IMPORTANT: reuse train scaler for val/test)
# -----------------------------
# If your Dataset classes accept scaler=... (recommended), this will reuse it.
# If they don't accept scaler, remove the scaler=... arguments.
train_set = Data(
    root_path=data_info['root_path'], flag='train',
    seq_len=args.seq_length, pre_len=args.pre_length,
    type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    scaler=None
)
train_scaler = getattr(train_set, "scaler", None)

val_set = Data(
    root_path=data_info['root_path'], flag='val',
    seq_len=args.seq_length, pre_len=args.pre_length,
    type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    scaler=train_scaler
)

test_set = Data(
    root_path=data_info['root_path'], flag='test',
    seq_len=args.seq_length, pre_len=args.pre_length,
    type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    scaler=train_scaler
)

train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=True)
val_dataloader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
test_dataloader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

print("Train samples:", len(train_set))
print("Val samples:", len(val_set))
print("Test samples:", len(test_set))


MODELS_SET2 = ["TSGRU", "TSTCN", "TSFormer", "SpikformerCPG", "SpikeGRU", "SpikeRNN", "SpikeTCN"]


# -----------------------------
# Validate/Test (print both metrics)
# -----------------------------
def validate(model, vali_loader, scaler):
    model.eval()
    cnt = 0
    loss_total = 0.0
    preds_list = []
    trues_list = []

    for x, y in vali_loader:
        if args.model in MODELS_SET2 and args.model != 'TSGRU':
            reset_states(model=model)
        elif args.model == 'TSGRU':
            remove(model=model.net[0].tslif)

        x = x.float().to(args.device)
        y = y.float().to(args.device)

        forecast, _ = model(x)
        if len(forecast.shape) == 4:
            forecast = forecast.mean(dim=0)

        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        cnt += 1

        if args.model not in MODELS_SET2:
            functional.reset_net(model)

        preds_list.append(forecast.detach().cpu().numpy())
        trues_list.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    score_scaled, score_orig = compute_scores_scaled_and_orig(trues, preds, scaler)

    print(_fmt_score("SCALED", score_scaled))
    print(_fmt_score("ORIG  ", score_orig))

    model.train()
    return loss_total / max(1, cnt)


def test(model, result_test_file, scaler, load_epoch=97):
    model = load_model_ts(model, result_test_file, load_epoch)
    model.eval()

    preds_list = []
    trues_list = []

    for x, y in test_dataloader:
        if args.model in MODELS_SET2 and args.model != 'TSGRU':
            reset_states(model=model)
        elif args.model == 'TSGRU':
            remove(model=model.net[0].tslif)

        x = x.float().to(args.device)
        y = y.float().to(args.device)

        forecast, _ = model(x)
        if len(forecast.shape) == 4:
            forecast = forecast.mean(dim=0)

        if args.model not in MODELS_SET2:
            functional.reset_net(model)

        preds_list.append(forecast.detach().cpu().numpy())
        trues_list.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    score_scaled, score_orig = compute_scores_scaled_and_orig(trues, preds, scaler)

    print(_fmt_score("SCALED", score_scaled))
    print(_fmt_score("ORIG  ", score_orig))

    return score_scaled, score_orig


# -----------------------------
# Optim/scheduler builder
# -----------------------------
def build_opt_sched(model, lr=3e-4, wd=0.01, gate_lr_ratio=0.3,
                    warmup_epochs=8, total_epochs=100):
    decay, no_decay, gate = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_l = name.lower()
        is_bias = name.endswith('bias')
        is_norm = ('norm' in name_l) or ('bn' in name_l)
        is_embed = ('embeddings' in name_l) or ('time_basis' in name_l)
        if 'freq_gate' in name_l and 'log_alpha' in name_l:
            gate.append(p)
        elif is_bias or is_norm or is_embed or p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)

    optim = torch.optim.AdamW([
        {'params': decay,     'lr': lr,                'weight_decay': wd},
        {'params': no_decay,  'lr': lr,                'weight_decay': 0.0},
        {'params': gate,      'lr': lr * gate_lr_ratio,'weight_decay': 0.0},
    ], betas=(0.9, 0.99), eps=1e-8)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, total_epochs - warmup_epochs), eta_min=lr * 0.1
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    return optim, sched


# -----------------------------
# Main (5 runs)
# -----------------------------
if __name__ == '__main__':
    ei_target = None

    seeds = [2021, 2022, 2023, 2024, 2025]

    # Store both SCALED and ORIG results
    scaled_results = {'mape': [], 'mae': [], 'rmse': [], 'r2': [], 'rse': []}
    orig_results   = {'mape': [], 'mae': [], 'rmse': [], 'r2': [], 'rse': []}

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Starting Run {run_idx + 1}/5 | seed={seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        result_train_file = os.path.join('output', args.data, args.model, f'train_run_{run_idx+1}_seed_{seed}')
        result_test_file  = os.path.join('output', args.data, args.model, f'train_run_{run_idx+1}_seed_{seed}')
        os.makedirs(result_train_file, exist_ok=True)
        os.makedirs(result_test_file,  exist_ok=True)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Model init
        if args.model == 'SpikF_GO':
            model = SpikF_GO(args, pre_length=args.pre_length, embed_size=args.embed_size,
                             feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim, my_lr_scheduler = build_opt_sched(
                model, lr=args.learning_rate, wd=0.01,
                warmup_epochs=max(4, args.train_epochs//8), total_epochs=args.train_epochs
            )
        elif args.model == 'SpikF_GO1':
            model = SpikF_GO1(args, pre_length=args.pre_length, embed_size=args.embed_size,
                              feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim, my_lr_scheduler = build_opt_sched(
                model, lr=args.learning_rate, wd=0.01,
                warmup_epochs=max(4, args.train_epochs//8), total_epochs=args.train_epochs
            )
        elif args.model == 'SpikF_GO2':
            model = SpikF_GO2(args, pre_length=args.pre_length, embed_size=args.embed_size,
                              feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim, my_lr_scheduler = build_opt_sched(
                model, lr=args.learning_rate, wd=0.01,
                warmup_epochs=max(4, args.train_epochs//8), total_epochs=args.train_epochs
            )
        elif args.model == 'SpikF_GO1_CPG':
            model = SpikF_GO1_CPG(args, pre_length=args.pre_length, embed_size=args.embed_size,
                                  feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim, my_lr_scheduler = build_opt_sched(
                model, lr=args.learning_rate, wd=0.01,
                warmup_epochs=max(4, args.train_epochs//8), total_epochs=args.train_epochs
            )
        elif args.model == 'SpikF_GO2_CPG':
            model = SpikF_GO2_CPG(args, pre_length=args.pre_length, embed_size=args.embed_size,
                                  feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim, my_lr_scheduler = build_opt_sched(
                model, lr=args.learning_rate, wd=0.01,
                warmup_epochs=max(4, args.train_epochs//8), total_epochs=args.train_epochs
            )
        elif args.model == 'FGN':
            model = FGN(args, pre_length=args.pre_length, embed_size=args.embed_size,
                        feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'SpikF':
            model = SpikF(args, input_len=args.seq_length, patch_num=args.patch_num, patch_dim=args.patch_dim,
                          T=args.T, blocks=args.blocks, D=args.feature_size, pred_len=args.pre_length,
                          tau=args.tau, alpha=args.alpha, hidden_dim=args.hidden_size)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'iSpikformer':
            model = iSpikformer(args, input_len=args.seq_length, patch_num=args.patch_num, patch_dim=args.patch_dim,
                                T=args.T, blocks=args.blocks, D=args.feature_size, pred_len=args.pre_length,
                                tau=args.tau, alpha=args.alpha, hidden_dim=args.hidden_size)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'SpikF_GO_CPG':
            model = SpikF_GO_CPG(args, pre_length=args.pre_length, embed_size=args.embed_size,
                                 feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
            my_optim, my_lr_scheduler = build_opt_sched(
                model, lr=args.learning_rate, wd=0.01,
                warmup_epochs=max(4, args.train_epochs//8), total_epochs=args.train_epochs
            )
        elif args.model == 'TSGRU':
            model = TSGRU(args, hidden_size=args.hidden_size, layers=args.blocks,
                         num_steps=args.T, input_size=args.feature_size)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'TSTCN':
            model = TSTCN(args=args, num_levels=args.blocks)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'TSFormer':
            model = TSFormer(args=args)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'SpikformerCPG':
            model = SpikformerCPG(args=args)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'SpikeGRU':
            model = SpikeGRU(args, hidden_size=args.hidden_size, layers=args.blocks,
                             num_steps=args.T, input_size=args.feature_size)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'SpikeRNN':
            model = SpikeRNN(args, hidden_size=args.hidden_size, layers=args.blocks,
                             num_steps=args.T, input_size=args.feature_size)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        elif args.model == 'SpikeTCN':
            model = SpikeTCN(args=args, num_levels=args.blocks)
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        model = model.to(device)
        forecast_loss = nn.MSELoss(reduction='mean').to(device)

        # -------------------
        # Train
        # -------------------
        for epoch in range(args.train_epochs):
            warm = int(0.3 * args.train_epochs)
            cool = epoch >= warm

            epoch_start_time = time.time()
            model.train()
            loss_total = 0.0
            cnt = 0

            for x, y in train_dataloader:
                if args.model in MODELS_SET2 and args.model != 'TSGRU':
                    reset_states(model=model)
                elif args.model == 'TSGRU':
                    remove(model=model.net[0].tslif)

                x = x.float().to(device)
                y = y.float().to(device)

                forecast, aux = model(x)

                if len(forecast.shape) == 4:
                    y_rep = y.repeat(args.T, 1, 1, 1)
                else:
                    y_rep = y

                if (args.model in ['SpikF_GO', 'SpikF_GO_CPG', 'SpikF_GO1', 'SpikF_GO2', 'SpikF_GO1_CPG', 'SpikF_GO2_CPG']) and args.energy_loss:
                    with torch.no_grad():
                        cur_ei = (aux['enc_rate'].detach() * aux['rho_hat'].detach()).item()
                        if ei_target is None:
                            ei_target = cur_ei
                        else:
                            ei_target = 0.99 * ei_target + 0.01 * cur_ei
                    energy = aux['rho_hat']  # ok

                    low, high = 0.04, 0.18
                    rate = aux['enc_rate']
                    rate_loss = F.relu(low - rate).pow(2) + F.relu(rate - high).pow(2)

                    ei_loss = (aux['enc_rate'] * aux['rho_hat'] - ei_target) ** 2

                    energy_lambda = (1e-2 if not cool else 2e-2)
                    spike_lambda  = 1e-2        # <-- change from 1e-3
                    lambda_ei     = 0.0 if epoch < 5 else 1e-3  # <-- warm-up

                    loss = forecast_loss(forecast, y_rep) \
                        + energy_lambda * energy \
                        + spike_lambda * rate_loss \
                        + lambda_ei * ei_loss
                else:
                    loss = forecast_loss(forecast, y_rep)

                my_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                my_optim.step()

                loss_total += float(loss)
                cnt += 1

                if args.model not in MODELS_SET2:
                    functional.reset_net(model)

            if (epoch + 1) % args.exponential_decay_step == 0:
                my_lr_scheduler.step()

            if (epoch + 1) % args.validate_freq == 0:
                val_loss = validate(model, val_dataloader, train_scaler)
                enc_rate_v = float(aux.get('enc_rate', torch.tensor(0.0)))
                gate_l0_v  = float(aux.get('rho_hat', torch.tensor(0.0)))
                rho_v      = float(aux.get('rho_mean', torch.tensor(0.0)))
                freq_act_v = float(aux.get('freq_mask_active', torch.tensor(0.0)))

                print('Run {} | epoch {:03d} | {:5.2f}s | train_loss {:5.4f} | val_loss {:5.4f} | enc_rate {:.3f} | gate_L0 {:.3f} | rho {:.3f} | f_active {:.3f}'.format(
                    run_idx + 1, epoch, (time.time() - epoch_start_time), loss_total / max(1, cnt), val_loss,
                    enc_rate_v, gate_l0_v, rho_v, freq_act_v))

            if (epoch + 1) % 49 == 0:
                save_model_ts(model, result_train_file, epoch)

        save_model_ts(model, result_train_file, f'final_run_{run_idx+1}')

        # -------------------
        # Test (both metrics)
        # -------------------
        print("=== TEST ===")
        score_scaled, score_orig = test(model, result_test_file, train_scaler, load_epoch=97)

        scaled_results['mape'].append(score_scaled[0])
        scaled_results['mae'].append(score_scaled[1])
        scaled_results['rmse'].append(score_scaled[2])
        scaled_results['r2'].append(score_scaled[3])
        scaled_results['rse'].append(score_scaled[4])

        orig_results['mape'].append(score_orig[0])
        orig_results['mae'].append(score_orig[1])
        orig_results['rmse'].append(score_orig[2])
        orig_results['r2'].append(score_orig[3])
        orig_results['rse'].append(score_orig[4])

        print(f"Run {run_idx + 1} completed.")
        print(_fmt_score("SCALED", score_scaled))
        print(_fmt_score("ORIG  ", score_orig))

    # -------------------
    # Summary across runs
    # -------------------
    def _mean_std(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    print(f"\n{'='*60}")
    print("FINAL RESULTS ACROSS 5 RUNS (SCALED + ORIG)")
    print(f"{'='*60}")

    for tag, store in [("SCALED", scaled_results), ("ORIG", orig_results)]:
        mape_pct = np.asarray(store['mape'], dtype=np.float64) * 100.0
        m_mean, m_std = _mean_std(mape_pct)
        a_mean, a_std = _mean_std(store['mae'])
        r_mean, r_std = _mean_std(store['rmse'])
        r2_mean, r2_std = _mean_std(store['r2'])
        rse_mean, rse_std = _mean_std(store['rse'])

        print(f"\n[{tag}]")
        print(f"MAPE: {mape_pct}  | mean={m_mean:.6f} std={m_std:.6f}")
        print(f"MAE : {np.array(store['mae'])}   | mean={a_mean:.6f} std={a_std:.6f}")
        print(f"RMSE: {np.array(store['rmse'])}  | mean={r_mean:.6f} std={r_std:.6f}")
        print(f"R2  : {np.array(store['r2'])}    | mean={r2_mean:.6f} std={r2_std:.6f}")
        print(f"RSE  : {np.array(store['rse'])}    | mean={rse_mean:.6f} std={rse_std:.6f}")

    # Save summary (scaled only, MAPE in percent units)
    summary_file = os.path.join('output', args.data, args.model, 'summary_results_scaled.txt')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    with open(summary_file, 'w') as f:
        f.write("Results across 5 runs (SCALED only, MAPE in percent):\n")
        f.write(f"Seeds used: {seeds}\n\n")

        for tag, store in [("SCALED", scaled_results)]:
            mape_pct = np.asarray(store['mape'], dtype=np.float64) * 100.0
            m_mean, m_std = _mean_std(mape_pct)
            a_mean, a_std = _mean_std(store['mae'])
            r_mean, r_std = _mean_std(store['rmse'])
            r2_mean, r2_std = _mean_std(store['r2'])
            rse_mean, rse_std = _mean_std(store['rse'])

            f.write(f"[{tag}]\n")
            f.write(f"MAPE - Individual: {mape_pct}\n")
            f.write(f"MAPE - Mean: {m_mean:.6f}, Std: {m_std:.6f}\n")
            f.write(f"MAE  - Individual: {np.array(store['mae'])}\n")
            f.write(f"MAE  - Mean: {a_mean:.6f}, Std: {a_std:.6f}\n")
            f.write(f"RMSE - Individual: {np.array(store['rmse'])}\n")
            f.write(f"RMSE - Mean: {r_mean:.6f}, Std: {r_std:.6f}\n")
            f.write(f"R2   - Individual: {np.array(store['r2'])}\n")
            f.write(f"R2   - Mean: {r2_mean:.6f}, Std: {r2_std:.6f}\n\n")
            f.write(f"RSE   - Individual: {np.array(store['rse'])}\n")
            f.write(f"RSE   - Mean: {rse_mean:.6f}, Std: {rse_std:.6f}\n\n")

    print(f"\nSaved summary to: {summary_file}")
