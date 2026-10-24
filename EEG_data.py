import torch
import re
from torch.utils.data import TensorDataset
import numpy as np


def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."""
    reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)

    return subject_files


def load_data(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate


def load_dual_data(fpz_files, oz_files):
    """加载双通道数据"""
    fpz_data = []
    oz_data = []
    labels = []

    for fpz_path, oz_path in zip(fpz_files, oz_files):
        fpz = np.load(fpz_path)
        oz = np.load(oz_path)

        # 假设NPZ文件中'x'是信号，'y'是标签
        fpz_data.append(fpz['x'])
        oz_data.append(oz['x'])
        labels.append(fpz['y'])  # 两个文件的标签应该相同

    return np.concatenate(fpz_data), np.concatenate(oz_data), np.concatenate(labels)


def load_and_process_data(dataset_path, all_files, subject_ids):
    """统一加载和处理数据"""
    files = np.hstack([get_subject_files(dataset_path, all_files, sid) for sid in subject_ids])
    x, y, _ = load_data(files)
    return np.concatenate(x), np.concatenate(y)


def create_tensor_dataset(x_cz, x_oz, y, device):
    """创建正确的双通道数据集"""
    # 确保输入是2D数组 [n_samples, 3000]
    x_cz = np.array([x.reshape(-1) for x in x_cz]) if isinstance(x_cz, list) else x_cz.reshape(len(x_cz), -1)
    x_oz = np.array([x.reshape(-1) for x in x_oz]) if isinstance(x_oz, list) else x_oz.reshape(len(x_oz), -1)

    # 合并为3D数组 [n_samples, 2, 3000]
    merged_x = np.stack([x_cz, x_oz], axis=1)

    # 转换为张量并确保没有额外维度
    return TensorDataset(
        torch.tensor(merged_x, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device)
    )

def create_multi_tensor_dataset(x_cz, x_oz, x_eog, y, device):
    """创建正确的双通道数据集"""
    # 确保输入是2D数组 [n_samples, 3000]
    x_cz = np.array([x.reshape(-1) for x in x_cz]) if isinstance(x_cz, list) else x_cz.reshape(len(x_cz), -1)
    x_oz = np.array([x.reshape(-1) for x in x_oz]) if isinstance(x_oz, list) else x_oz.reshape(len(x_oz), -1)
    x_eog = np.array([x.reshape(-1) for x in x_eog]) if isinstance(x_eog, list) else x_eog.reshape(len(x_eog), -1)

    # 合并为3D数组 [n_samples, 2, 3000]
    merged_x = np.stack([x_cz, x_oz, x_eog], axis=1)

    # 转换为张量并确保没有额外维度
    return TensorDataset(
        torch.tensor(merged_x, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device)
    )

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick(dataset, label: int, n: int) -> list:
    """Return up to *n* signal arrays whose label matches *label*."""
    collected = []
    for signal, lbl in dataset:
        if int(lbl) == label:
            collected.append(signal)
        if len(collected) >= n:
            break
    return collected


def _draw_channels(ax, signal, colors: list[str], channel_names: list[str]) -> None:
    """
    Plot EEG-1, EEG-2, and EOG as three stacked traces on a single Axes.

    Parameters
    ----------
    signal        : array-like, shape (C, T) — C channels, T time steps.
                    If shape is (T,) it is treated as a single channel.
    colors        : list of hex/named colours, one per channel.
    channel_names : list of channel labels, one per channel.
    """
    if hasattr(signal, "cpu"):
        signal = signal.cpu()
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]          # (1, T)

    n_channels = signal.shape[0]
    t = np.arange(signal.shape[1])

    # Vertical offset so channels don't overlap
    offsets = []
    for c in range(n_channels):
        rng = signal[c].max() - signal[c].min() + 1e-9
        offsets.append(c * rng * 1.4)

    for c in range(n_channels):
        ax.plot(
            t,
            signal[c] + offsets[c],
            color=colors[c % len(colors)],
            linewidth=0.8,
            label=channel_names[c] if c < len(channel_names) else f"Ch{c}",
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(
        channel_names[:n_channels],
        fontsize=7,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(y=0, color="none")   # keep y-range stable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_gan_samples(
    train_dataset,
    augmented_train_dataset,
    label_n1: int = 1,
    n_samples: int = 5,
    save_path: str = "sleepgan_comparison.png",
    show: bool = False,
) -> None:
    """
    Randomly draw *n_samples* N1 examples from the GAN-augmented dataset and
    the original dataset, then produce a side-by-side comparison figure.

    Each subplot shows three stacked traces: EEG-1, EEG-2, and EOG.

    Parameters
    ----------
    train_dataset           : Original training dataset (real samples).
    augmented_train_dataset : SleepGAN-augmented dataset.
    label_n1                : Integer label for sleep stage N1 (default 1).
    n_samples               : Number of samples per row (default 5).
    save_path               : Output file path (default "sleepgan_comparison.png").
    show                    : Call plt.show() after saving; set False on servers.
    channel_names           : Names for the C channels in each signal.
                              Defaults to ["EEG-1", "EEG-2", "EOG"].
    """
    channel_names = ["EEG-1", "EEG-2", "EOG"]

    # Colour palette — one hue per channel, two brightness levels (gen / real)
    CHANNEL_COLORS_GEN  = ["#5A7FBB", "#6AADBB", "#7ABBA0"]   # cool tones
    CHANNEL_COLORS_REAL = ["#E07050", "#E09050", "#C07080"]    # warm tones

    gan_samples  = _pick(augmented_train_dataset, label_n1, n_samples)
    real_samples = _pick(train_dataset,           label_n1, n_samples)

    n = min(len(gan_samples), len(real_samples), n_samples)
    if n == 0:
        print("[visualize_gan_samples] No N1 samples found — skipping plot.")
        return

    fig, axes = plt.subplots(
        2, n,
        figsize=(3 * n, 5),
        constrained_layout=True,
    )
    # Ensure axes is always 2-D even when n == 1
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("N1 Sample Comparison: Generated (top) vs Real (bottom)", fontsize=13)

    rows = [
        (gan_samples,  CHANNEL_COLORS_GEN,  "Generated\n(SleepGAN)"),
        (real_samples, CHANNEL_COLORS_REAL, "Real"),
    ]

    for row_idx, (samples, colors, row_label) in enumerate(rows):
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            _draw_channels(ax, samples[col_idx], colors, channel_names)

            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9, color=colors[0])
            if row_idx == 0:
                ax.set_title(f"Sample {col_idx + 1}", fontsize=9)

    # Shared legend (one entry per channel) in the top-right corner
    legend_handles = [
        plt.Line2D([0], [0], color=CHANNEL_COLORS_GEN[i], linewidth=1.2,
                   label=channel_names[i])
        for i in range(len(channel_names))
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        title="Channels",
        title_fontsize=8,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

