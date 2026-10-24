import os, glob, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from EEG_data import load_and_process_data, create_multi_tensor_dataset
from utils_eeg import load_seq_ids
from diffusion import balance_with_diffu
from EEG_visualize import visualize_gan_samples

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset_cz  = "~/sleepedf/EDF_78/fpz_cz"
dataset_oz  = "~/sleepedf/EDF_78/pz_oz"
dataset_eog = "~/sleepedf/EDF_78/eog"
cz_files    = glob.glob(os.path.join(dataset_cz,  "*.npz"))
oz_files    = glob.glob(os.path.join(dataset_oz,  "*.npz"))
eog_files   = glob.glob(os.path.join(dataset_eog, "*.npz"))
seq_sids    = load_seq_ids("~/sleepedf/EDF_78/sleepedf.txt")

def make_dataset(sids):
    d = {k: load_and_process_data(ds, f, sids)
         for k, ds, f in [("cz", dataset_cz, cz_files),
                           ("oz", dataset_oz, oz_files),
                           ("eog", dataset_eog, eog_files)]}
    return create_multi_tensor_dataset(d["cz"][0], d["oz"][0], d["eog"][0], d["cz"][1], device)

os.makedirs("diffu_data", exist_ok=True)
os.makedirs("diffu_Figure",   exist_ok=True)

for fold_idx in range(10):
    print(f"===== Fold {fold_idx} =====")

    test_sids  = np.array_split(seq_sids, 10)[fold_idx]
    train_sids = np.setdiff1d(seq_sids, test_sids)

    train_full = make_dataset(train_sids)
    train_size = int(0.9 * len(train_full))
    train_dataset, _ = random_split(train_full, [train_size, len(train_full) - train_size])

    gan_only = balance_with_diffu(train_dataset, device)

    # 保存 GAN 样本
    X = np.stack([s[0].cpu().numpy() if torch.is_tensor(s[0]) else s[0] for s in gan_only])
    y = np.array([s[1].cpu().item() if torch.is_tensor(s[1]) else int(s[1]) for s in gan_only])
    np.savez(f"diffu_data/diffu{fold_idx}.npz", X=X, y=y)

    visualize_gan_samples(train_dataset, gan_only, label_n1=1, n_samples=5,
                          save_path=f"diffu_Figure/diffu{fold_idx}.png")
    print("*" * 60)