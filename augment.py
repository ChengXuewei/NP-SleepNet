"""
augment.py
针对三通道EEG/EOG信号(Fpz-Cz, Pz-Oz, EOG)的N1类数据增强。
每个样本随机选择"一个"通道进行增强(该通道随机叠加2种增强)，
其余通道保持不变，避免破坏通道间的特征差异。

输入: signal, shape = (B, 3, L) 或 (3, L)
输出: 同shape
"""

import torch
import torch.nn.functional as F


AUG_TYPES = ['noise', 'shift', 'scale', 'filter', 'warp', 'freq', 'crop']


def _augment_single_channel(x: torch.Tensor) -> torch.Tensor:
    """
    对单通道信号 (1, 1, L) 施加随机2种增强组合。
    """
    device = x.device
    orig_len = x.shape[-1]
    augmented = x.clone()

    selected_augs = torch.randperm(len(AUG_TYPES))[:2]

    for aug_idx in selected_augs:
        aug = AUG_TYPES[aug_idx]

        if aug == 'noise':
            noise = torch.randn_like(augmented) * 0.01 * augmented.std()
            augmented = augmented + noise

        elif aug == 'shift':
            shift = torch.randint(-50, 51, (1,)).item()
            augmented = torch.roll(augmented, shifts=shift, dims=-1)
            if shift > 0:
                boundary_value = augmented[..., shift:shift + 1]
                augmented[..., :shift] = boundary_value.expand_as(augmented[..., :shift])
            elif shift < 0:
                s = abs(shift)
                if s < orig_len:
                    boundary_value = augmented[..., -s - 1:-s]
                    augmented[..., -s:] = boundary_value.expand_as(augmented[..., -s:])

        elif aug == 'scale':
            scale = torch.empty(1, device=device).uniform_(0.9, 1.1)
            augmented = augmented * scale

        elif aug == 'filter':
            kernel_size = 5
            weights = torch.rand(kernel_size, device=device) - 0.5
            weights = weights / weights.sum()
            augmented = F.conv1d(augmented, weights.view(1, 1, -1), padding='same')

        elif aug == 'warp':
            new_len = max(2, int(orig_len * torch.empty(1).uniform_(0.9, 1.1).item()))
            augmented = F.interpolate(augmented, size=new_len, mode='linear', align_corners=True)
            augmented = F.interpolate(augmented, size=orig_len, mode='linear', align_corners=True)

        elif aug == 'freq':
            fft_signal = torch.fft.fft(augmented)
            phase_shift = torch.empty_like(fft_signal.real).uniform_(-0.1, 0.1)
            fft_signal = fft_signal * torch.exp(1j * phase_shift)
            augmented = torch.fft.ifft(fft_signal).real

        elif aug == 'crop':
            crop_ratio = torch.empty(1).uniform_(0.9, 1.0).item()
            crop_size = max(2, int(orig_len * crop_ratio))
            max_start = orig_len - crop_size
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            cropped = augmented[..., start:start + crop_size]
            augmented = F.interpolate(cropped, size=orig_len, mode='linear', align_corners=True)

    return augmented


def augment_N1(signal: torch.Tensor) -> torch.Tensor:
    """
    三通道N1数据增强入口。

    Args:
        signal: (B, 3, L) 或 (3, L)，三通道分别对应如
                [Fpz-Cz EEG, Pz-Oz EEG, EOG]

    Returns:
        与输入相同shape的增强后信号；每个样本仅随机选取1个通道做增强，
        其他通道保持原值不变。
    """
    squeeze_batch = False
    if signal.dim() == 2:  # (3, L) -> (1, 3, L)
        signal = signal.unsqueeze(0)
        squeeze_batch = True
    elif signal.dim() != 3:
        raise ValueError(f"Expected signal of shape (B,3,L) or (3,L), got {signal.shape}")

    B, C, L = signal.shape
    out = signal.clone()

    for b in range(B):
        ch = torch.randint(0, C, (1,)).item()  # 随机选择一个通道
        single = out[b, ch].unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        aug_single = _augment_single_channel(single)
        out[b, ch] = aug_single.squeeze(0).squeeze(0)

    if squeeze_batch:
        out = out.squeeze(0)

    return out
