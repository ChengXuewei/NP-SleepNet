import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_random(dataset, label: int, n: int, rng: np.random.Generator) -> list:
    """Randomly select up to *n* signal arrays whose label matches *label*.

    Unlike the old `_pick`, this scans the whole dataset first and then
    samples without replacement, so you actually get a different set of
    samples each call instead of always the first n in dataset order.
    """
    matches = [signal for signal, lbl in dataset if int(lbl) == label]
    if not matches:
        return []
    n = min(n, len(matches))
    idx = rng.choice(len(matches), size=n, replace=False)
    return [matches[i] for i in idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_gan_samples(
    real_dataset,         # iterable of (signal, label) pairs
    fake_dataset,         # iterable of (signal, label) pairs
    label_n1=1,
    n_samples=5,
    channel_names=("EEG-1", "EEG-2", "EOG"),
    real_colors=("#E07B54", "#E0A354", "#C1566B"),    # warm: real
    fake_colors=("#5B8DB8", "#6FC4C4", "#7DAF6E"),    # cool: generated
    figsize_per_cell=(3.0, 1.0),
    save_path="gan_comparison.png",
    seed=None,
):
    """
    Layout
    ------
    columns = sample index (1..n_samples)
    rows    = channel, doubled into a "Generated" block (top) and a
              "Real" block (bottom), separated by a thin spacer row.
    Each column pairs one randomly-drawn real sample with one randomly-drawn
    generated sample; each row is the same channel across both blocks.

    Why this version differs from the original
    --------------------------------------------
    The previous implementation created *extra* `fig.add_subplot` axes for
    column titles and for the rotated group labels, and those extra axes
    landed on the exact same GridSpec cells that already held plotted data
    (gs[0, si] for titles, gs[mid, 0] for labels). The new blank axes drew
    on top of the real ones, which is why some panels showed an empty
    default 0-1 box instead of the signal. This version reuses the axes
    that were already plotted (via the `axes` dict returned by
    `_plot_block`) for titles, and uses `fig.text` (not a new axes) for the
    rotated group labels, so nothing ever overlaps plotted data.
    """
    rng = np.random.default_rng(seed)

    def to_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu()
        x = np.asarray(x)
        return x[np.newaxis] if x.ndim == 1 else x   # (C, T)

    real_samples = [to_np(s) for s in _pick_random(real_dataset, label_n1, n_samples, rng)]
    fake_samples = [to_np(s) for s in _pick_random(fake_dataset, label_n1, n_samples, rng)]

    n_samples = min(len(real_samples), len(fake_samples))
    if n_samples == 0:
        raise ValueError(f"No samples found with label_n1={label_n1} in one of the datasets.")
    real_samples, fake_samples = real_samples[:n_samples], fake_samples[:n_samples]

    n_channels = real_samples[0].shape[0]
    n_rows = n_channels * 2 + 1   # fake block + spacer row + real block

    fig_w = figsize_per_cell[0] * n_samples
    fig_h = figsize_per_cell[1] * (n_rows - 1) + 0.8   # +0.8 for titles/labels

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    hr = [1] * n_channels + [0.15] + [1] * n_channels
    gs = gridspec.GridSpec(
        n_rows, n_samples,
        height_ratios=hr,
        hspace=0.08, wspace=0.06,
        left=0.12, right=0.97, top=0.90, bottom=0.03,
    )

    def _plot_block(block_start, samples, colors, names):
        axes = {}
        for ci in range(n_channels):
            row = block_start + ci
            all_vals = np.concatenate([s[ci] for s in samples])
            pad = (all_vals.max() - all_vals.min()) * 0.15 + 1e-9
            ylim = (all_vals.min() - pad, all_vals.max() + pad)

            for si, sig in enumerate(samples):
                ax = fig.add_subplot(gs[row, si])
                t = np.arange(sig.shape[1])
                ax.plot(t, sig[ci], color=colors[ci % len(colors)], linewidth=0.7)
                ax.set_ylim(ylim)
                ax.set_xlim(0, sig.shape[1] - 1)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                # ax.axhline(0, color="#cccccc", linewidth=0.4, zorder=0)

                if si == 0:
                    ax.set_ylabel(
                        names[ci], fontsize=7, rotation=0, labelpad=28,
                        va="center", ha="right", color=colors[ci % len(colors)],
                    )
                axes[(ci, si)] = ax
        return axes

    fake_axes = _plot_block(0, fake_samples, fake_colors, channel_names)
    real_axes = _plot_block(n_channels + 1, real_samples, real_colors, channel_names)

    # Column titles: set directly on the already-plotted top-row axes instead
    # of stacking a brand-new axes on the same cell.
    for si in range(n_samples):
        fake_axes[(0, si)].set_title(f"Sample {si + 1}", fontsize=9, fontweight="bold", pad=6)

    # Group labels: use fig.text positioned from the existing axes' bounding
    # boxes, so no new axes is created on top of plotted data.
    fig.canvas.draw()
    for axes_dict, label, color in [
        (fake_axes, "Generated\n(SleepDiffusion)", "#4a7aab"),
        (real_axes, "Real", "#c1566b"),
    ]:
        top_box = axes_dict[(0, 0)].get_position()
        bot_box = axes_dict[(n_channels - 1, 0)].get_position()
        y_mid = (top_box.y1 + bot_box.y0) / 2
        x_axes_left = top_box.x0
        x_text = x_axes_left - 0.02   
    
        fig.text(
            x_text, y_mid, label, fontsize=8, fontweight="bold",
            color=color, ha="center", va="center", rotation=90,
        )

    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig