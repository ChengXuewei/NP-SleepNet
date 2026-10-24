import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_roll(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Per-sample circular time-shift augmentation, vectorized via gather.
    x: (B, C, L). Each of the B samples gets its own random shift in
    [-max_shift, max_shift]. Cheap (~free on GPU), and a reasonable way to
    squeeze a bit more effective diversity out of small N1 calibration-free
    training sets without touching the physiology.
    """
    if max_shift <= 0:
        return x
    B, C, L = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    idx = (torch.arange(L, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % L
    idx = idx.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(x, 2, idx)


# ─── Hyperparameters ──────────────────────────────────────────────────────────
N_CHANNELS = 3
SEQ_LEN    = 3000
BATCH_SIZE = 32
FS         = 100
LABEL_N1   = 1

# --- v1 -> diffusion1: fix schedule / train-sample mismatch ----------------
# (unchanged from diffusion3 — see original derivation: T=200 + DDPM beta
# range under-noises the training distribution relative to what generate()
# samples from at inference; T=1000 fixes alpha_bar_T -> ~4e-5.)
T_DIFFUSION = 1000
BETA_START  = 1e-4
BETA_END    = 0.02
DDIM_STEPS  = 100
TIME_DIM    = 128

# --- diffusion3 -> diffusion4: bump backbone width for Sleep-EDF-78 -------
# Fold 0/1 logs on Sleep-EDF-78 show the eps/lf loss curves flattening by
# ~epoch 20 and barely moving over the remaining 160 epochs (eps
# 0.1131->0.1104, i.e. ~2.4% total improvement) while PSD log-corr lands at
# only 0.948-0.959 per channel -- markedly below the 0.99+ this same
# architecture reached on Sleep-EDF-20. A flat loss curve this early,
# combined with a spectral-fidelity regression, is the signature of a
# capacity ceiling rather than a schedule or data problem: Sleep-EDF-78
# brings ~7x the N1 sample count (17k-18k/fold vs ~2.3k-2.5k/fold) and a
# correspondingly much larger spread of inter-subject spectral profiles
# (different alpha/spindle frequencies, noise floors, EOG amplitude
# ranges), and CH=64 was sized for the smaller, more homogeneous
# Sleep-EDF-20 corpus. Widening the backbone (64 -> 96, ~2.25x parameter
# cost in the conv layers) gives the model enough capacity to represent
# that added heterogeneity instead of averaging over it.
# N_BLOCKS is left unchanged: RF = 12*(63+64+128) = 3060 already covers the
# full 3000-sample epoch, so more blocks would add depth/cost without
# addressing the actual bottleneck (width), and changing RF at the same
# time as width would confound the two effects in any ablation.
CH          = 96
N_BLOCKS    = 8             # dilations 1,2,4,8,16,32,64,128 -> RF ~= 3060 (full signal), unchanged

LR          = 2e-4
EPOCHS      = 180
EMA_DECAY   = 0.999
GRAD_CLIP_NORM = 1.0

# min-SNR-gamma loss weighting (Hang et al., 2023)
MIN_SNR_GAMMA = 5.0

# EOG (channel index 2) is the historically weak channel — keep extra
# weight on its noise-prediction loss and both spectral-shape losses below.
CH_LOSS_WEIGHT = [1.0, 1.0, 2.5]

# --- diffusion1 -> diffusion2: explicit low-frequency (slow-wave) loss -----
# (unchanged from diffusion3 — computed in eps-residual space to avoid the
# ~158x amplification from dividing by sqrt(alpha_bar_t) at high t.)
LOWFREQ_KERNEL      = 25
LOWFREQ_LOSS_WEIGHT = 0.15

# --- diffusion3 -> diffusion4: explicit high-frequency "texture" loss -----
# The low-freq loss above pushes the model to get slow/EOG-dominant content
# right; nothing in the loss explicitly protects the *other* end of the
# spectrum. On Sleep-EDF-78 several generated samples in the sanity plots
# show an unmistakable failure mode: near-perfectly regular, low-noise
# oscillations (e.g. the EOG rows and one EEG-1 row collapsing into a clean
# sinusoid) where real epochs are always irregular broadband signal. This
# is classic mode-averaging under a capacity/heterogeneity mismatch: faced
# with many plausible fine-grained noise textures across 78 subjects, the
# model regresses toward a smooth "average" waveform rather than committing
# to any one texture. Noise-MSE alone is necessarily an *average* over all
# frequencies too, so it under-penalizes this specific failure relative to
# how visually obvious/damaging it is.
# Fix: penalize the first-order time-difference (a cheap high-pass proxy)
# of eps_pred against the true noise's first-order difference, at the same
# per-timestep min-SNR weight as the other terms, in the same eps-residual
# space as the low-freq loss (same stability rationale: eps and noise are
# O(1) regardless of t, so no division blow-up). This is symmetric to the
# low-freq term and costs one extra elementwise diff + MSE per step.
HF_LOSS_WEIGHT = 0.08

# --- diffusion1 -> diffusion2: time-shift augmentation ---------------------
AUG_MAX_SHIFT = 150

# Sampling stochasticity for the generalized-DDIM update.
SAMPLE_ETA = 0.0


# =============================================================================
# Building blocks
# =============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Dilated residual conv block with GroupNorm + SiLU (full-context, non-causal)."""
    def __init__(self, ch: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Denoiser(nn.Module):
    """
    (x_t, t) -> predicted noise eps, shape (B, 3, 3000) -> (B, 3, 3000).
    Time step is embedded once and broadcast-added as a channel-wise bias.
    """
    def __init__(self, ch: int = CH, n_blocks: int = N_BLOCKS, time_dim: int = TIME_DIM):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, ch),
        )
        self.in_conv = nn.Conv1d(N_CHANNELS, ch, 7, padding=3)
        self.blocks = nn.ModuleList([
            ResBlock(ch, kernel_size=7, dilation=2 ** i) for i in range(n_blocks)
        ])
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, N_CHANNELS, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        t_bias = self.time_mlp(self.time_embed(t))[:, :, None]  # (B, ch, 1)
        h = h + t_bias
        for block in self.blocks:
            h = block(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# =============================================================================
# Sleepdiffu — same public interface as before:
#   Sleepdiffu(device).fit(x_n1) -> None
#   Sleepdiffu(device).generate(n_samples, use_ema=True) -> np.ndarray
# =============================================================================
class Sleepdiffu:
    def __init__(self, device: torch.device):
        self.device = device

        self.model = Denoiser().to(device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=EPOCHS)

        self.model_ema = Denoiser().to(device)
        self.model_ema.load_state_dict(self.model.state_dict())
        for p in self.model_ema.parameters():
            p.requires_grad_(False)

        betas          = torch.linspace(BETA_START, BETA_END, T_DIFFUSION)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod.to(device)

        # Kept for logging/diagnostics only.
        self.global_mean: np.ndarray = None
        self.global_std:  np.ndarray = None

        # Per-sample (mu, std) bank used to bootstrap realistic sample-level
        # amplitude diversity back into generated samples.
        self.sample_mu:  torch.Tensor = None   # (N, C)
        self.sample_std: torch.Tensor = None   # (N, C)

    @torch.no_grad()
    def _update_ema(self, decay: float = EMA_DECAY):
        for p_ema, p in zip(self.model_ema.parameters(), self.model.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for b_ema, b in zip(self.model_ema.buffers(), self.model.buffers()):
            b_ema.copy_(b)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        self.global_mean = x.mean(axis=(0, 2), keepdims=True)
        self.global_std  = x.std(axis=(0, 2), keepdims=True).clip(1e-6)

        mu  = x.mean(axis=2, keepdims=True)          # (N, C, 1)
        std = x.std(axis=2, keepdims=True).clip(1e-6)  # (N, C, 1)

        self.sample_mu  = torch.from_numpy(mu.squeeze(-1)).float()   # (N, C)
        self.sample_std = torch.from_numpy(std.squeeze(-1)).float()  # (N, C)

        return (x - mu) / std

    def fit(self, x_n1: np.ndarray):
        print(f"[SleepDiffusion] Training on {len(x_n1)} N1 samples  "
              f"(device={self.device})  T={T_DIFFUSION}  N_BLOCKS={N_BLOCKS}  CH={CH}")

        x_norm = self._normalize(x_n1).astype(np.float32)
        print(f"[SleepDiffusion] Per-sample normalised: "
              f"mean={x_norm.mean():.4f}  std={x_norm.std():.4f}")

        ds     = TensorDataset(torch.FloatTensor(x_norm))
        bs     = min(BATCH_SIZE, len(x_n1))
        loader = DataLoader(ds, batch_size=bs, shuffle=True,
                            drop_last=(len(x_n1) >= bs))

        ch_weight = torch.tensor(CH_LOSS_WEIGHT, device=self.device).view(1, N_CHANNELS, 1)
        lowfreq_pad = LOWFREQ_KERNEL // 2

        for ep in range(1, EPOCHS + 1):
            ep_loss, ep_loss_eps, ep_loss_lf, ep_loss_hf, n = 0.0, 0.0, 0.0, 0.0, 0
            for (x0,) in loader:
                x0 = x0.to(self.device)
                x0 = random_roll(x0, AUG_MAX_SHIFT)
                B  = x0.size(0)

                t = torch.randint(0, T_DIFFUSION, (B,), device=self.device)
                noise = torch.randn_like(x0)

                ac_t      = self.alphas_cumprod[t]
                sqrt_ac   = ac_t.sqrt().view(B, 1, 1)
                sqrt_1mac = (1 - ac_t).sqrt().view(B, 1, 1)
                x_t = sqrt_ac * x0 + sqrt_1mac * noise

                eps_pred = self.model(x_t, t)

                snr = ac_t / (1 - ac_t).clamp(min=1e-8)
                snr_weight = (snr.clamp(max=MIN_SNR_GAMMA) / snr).view(B, 1, 1)

                loss_eps = (snr_weight * (eps_pred - noise) ** 2 * ch_weight).mean()

                # Low-frequency shape term (diffusion3 fix: computed in
                # eps-residual space, never in x0-pred space — see rationale
                # above for why the latter blows up ~158x at high t).
                lf_pred = F.avg_pool1d(eps_pred, kernel_size=LOWFREQ_KERNEL, stride=1,
                                        padding=lowfreq_pad, count_include_pad=False)
                lf_real = F.avg_pool1d(noise, kernel_size=LOWFREQ_KERNEL, stride=1,
                                        padding=lowfreq_pad, count_include_pad=False)
                loss_lf = (snr_weight * (lf_pred - lf_real) ** 2 * ch_weight).mean()

                # --- diffusion3 -> diffusion4: high-frequency texture term ---
                # First-order time-difference as a cheap high-pass proxy,
                # same eps-residual space (no division, no instability).
                # snr_weight is (B,1,1) so it broadcasts fine against the
                # length-(L-1) diff tensors without needing to be resliced.
                hf_pred = eps_pred[:, :, 1:] - eps_pred[:, :, :-1]
                hf_real = noise[:, :, 1:] - noise[:, :, :-1]
                loss_hf = (snr_weight * (hf_pred - hf_real) ** 2 * ch_weight).mean()

                loss = loss_eps + LOWFREQ_LOSS_WEIGHT * loss_lf + HF_LOSS_WEIGHT * loss_hf

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.opt.step()
                self._update_ema()

                ep_loss     += loss.item()
                ep_loss_eps += loss_eps.item()
                ep_loss_lf  += loss_lf.item()
                ep_loss_hf  += loss_hf.item()
                n += 1

            self.sched.step()

            if ep % 20 == 0 or ep == 1:
                lr_now = self.opt.param_groups[0]["lr"]
                print(f"  ep {ep:4d}/{EPOCHS}  loss={ep_loss / max(n, 1):.4f}  "
                      f"(eps={ep_loss_eps / max(n, 1):.4f}  "
                      f"lf={ep_loss_lf / max(n, 1):.4f}  "
                      f"hf={ep_loss_hf / max(n, 1):.4f})  lr={lr_now:.2e}")

        print("[SleepDiffusion] Training complete.")

    @torch.no_grad()
    def generate(self, n_samples: int, use_ema: bool = True,
                 eta: float = SAMPLE_ETA) -> np.ndarray:
        model = self.model_ema if use_ema else self.model
        model.eval()

        step_indices = np.unique(
            np.linspace(0, T_DIFFUSION - 1, DDIM_STEPS).round().astype(int)
        )[::-1]

        batches, done = [], 0
        while done < n_samples:
            B = min(BATCH_SIZE, n_samples - done)
            x = torch.randn(B, N_CHANNELS, SEQ_LEN, device=self.device)

            for i, t_cur in enumerate(step_indices):
                t_batch = torch.full((B,), int(t_cur), device=self.device, dtype=torch.long)
                eps = model(x, t_batch)

                ac_t    = self.alphas_cumprod[t_cur]
                x0_pred = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()

                if i + 1 < len(step_indices):
                    ac_next = self.alphas_cumprod[int(step_indices[i + 1])]
                    sigma = eta * torch.sqrt((1 - ac_next) / (1 - ac_t)) * \
                            torch.sqrt((1 - ac_t / ac_next).clamp(min=0.0))
                    dir_coeff = (1 - ac_next - sigma ** 2).clamp(min=0.0).sqrt()
                    x = ac_next.sqrt() * x0_pred + dir_coeff * eps
                    if eta > 0:
                        x = x + sigma * torch.randn_like(x)
                else:
                    x = x0_pred

                    mu_int  = x.mean(dim=2, keepdim=True)
                    std_int = x.std(dim=2, keepdim=True).clamp(min=1e-6)
                    x_shape = (x - mu_int) / std_int

                    idx = np.random.randint(0, self.sample_mu.shape[0], size=B)
                    mu  = self.sample_mu[idx].to(self.device).view(B, N_CHANNELS, 1)
                    std = self.sample_std[idx].to(self.device).view(B, N_CHANNELS, 1)
                    x = x_shape * std + mu

            batches.append(x.cpu().numpy())
            done += B

        model.train()
        x_gen = np.concatenate(batches, axis=0)[:n_samples]
        return x_gen


# =============================================================================
# Validation (unchanged)
# =============================================================================
def validate_generation(x_real: np.ndarray,
                        x_fake: np.ndarray,
                        fs: int = FS) -> dict:
    try:
        from scipy.signal import welch
    except ImportError:
        print("[validate] scipy not available — skipping.")
        return {}

    ch_names = ["Fpz-Cz", "Pz-Oz", "EOG"]
    results  = {}

    results["mean_real"] = float(x_real.mean())
    results["mean_fake"] = float(x_fake.mean())
    results["std_real"]  = float(x_real.std())
    results["std_fake"]  = float(x_fake.std())
    std_ratio = results["std_fake"] / (results["std_real"] + 1e-8)
    results["std_ratio"] = std_ratio

    print("\n[validate_generation]")
    print(f"  real  mean={results['mean_real']:.3f}  std={results['std_real']:.3f}")
    print(f"  fake  mean={results['mean_fake']:.3f}  std={results['std_fake']:.3f}")
    if not (0.7 < std_ratio < 1.3):
        print(f"  [WARN] std_ratio={std_ratio:.3f}  (target 0.7-1.3)")

    dc_real = x_real.mean(axis=(0, 2))
    dc_fake = x_fake.mean(axis=(0, 2))
    dc_diff = np.abs(dc_fake - dc_real)
    for c, name in enumerate(ch_names):
        results[f"dc_diff_ch{c}"] = float(dc_diff[c])
        if dc_diff[c] >= 1.0:
            print(f"  [WARN] DC offset {name} = {dc_diff[c]:.3f}  (target < 1.0)")

    for c, name in enumerate(ch_names):
        r = float(x_fake[:, c, :].std() / (x_real[:, c, :].std() + 1e-8))
        results[f"std_ratio_ch{c}"] = r
        if not (0.7 < r < 1.3):
            print(f"  [WARN] std_ratio {name} = {r:.3f}  (target 0.7-1.3)")

    print("  PSD log-corr:", end="")
    for ch in range(x_real.shape[1]):
        _, psd_r = welch(x_real[:, ch, :].flatten(), fs=fs, nperseg=256)
        _, psd_f = welch(x_fake[:, ch, :].flatten(), fs=fs, nperseg=256)
        corr = float(np.corrcoef(np.log(psd_r + 1e-8),
                                 np.log(psd_f + 1e-8))[0, 1])
        results[f"psd_corr_ch{ch}"] = corr
        print(f"  {ch_names[ch]}={corr:.4f}", end="")
    print("\n  (target > 0.95 per channel)")

    real_sample_std = x_real.std(axis=2).mean(axis=1)
    fake_sample_std = x_fake.std(axis=2).mean(axis=1)
    print(f"  per-sample std spread: real sd={real_sample_std.std():.3f}  "
          f"fake sd={fake_sample_std.std():.3f}  "
          f"(want fake close to real, not near 0)")
    results["sample_std_spread_real"] = float(real_sample_std.std())
    results["sample_std_spread_fake"] = float(fake_sample_std.std())

    return results


# =============================================================================
# Integration (unchanged)
# =============================================================================
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def balance_with_diffu(train_dataset,
                       device,
                       label_n1: int = LABEL_N1,
                       run_validation: bool = True):
    set_seed(42)

    all_data = [train_dataset[i] for i in range(len(train_dataset))]
    labels   = np.array([
        item[1].item() if torch.is_tensor(item[1]) else int(item[1])
        for item in all_data
    ])

    n1_mask    = labels == label_n1
    n_n1       = int(n1_mask.sum())
    n_non_n1   = int((~n1_mask).sum())
    n_generate = n_n1
    print(f"[Balance] N1={n_n1}, Non-N1={n_non_n1}, to generate={n_generate}")

    if n_generate <= 0:
        print("[Balance] Already balanced — skipping diffusion training.")
        return CombinedDataset(all_data), CombinedDataset([]), None

    x_n1 = np.stack([all_data[i][0].cpu().numpy()
                     for i in np.where(n1_mask)[0]])

    gan = Sleepdiffu(device)
    gan.fit(x_n1)

    x_syn = gan.generate(n_generate)

    if run_validation:
        validate_generation(x_n1, x_syn)

    x_syn_t      = torch.FloatTensor(x_syn).to(device)
    sample_label = all_data[np.where(n1_mask)[0][0]][1]
    syn_label    = (sample_label.clone() if torch.is_tensor(sample_label)
                    else torch.tensor(sample_label))
    syn_samples  = [(x_syn_t[i], syn_label) for i in range(n_generate)]

    combined = all_data + syn_samples
    print(f"[Balance] Final: {len(combined)} samples "
          f"(original {len(all_data)} + synthetic {n_generate})")

    return CombinedDataset(syn_samples)