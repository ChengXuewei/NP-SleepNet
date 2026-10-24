import os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset, random_split, ConcatDataset
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, accuracy_score
from EEG_data import load_and_process_data, create_multi_tensor_dataset
from utils_eeg import load_seq_ids
from augment import augment_N1
from trainer1 import train, calibrate_threshold_ensemble, np_predict_ensemble
from model import SleepStageModel

N1_LABEL = 1
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CombinedDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

dataset_cz = "~/sleepedf/EDF_20/fpz_cz"
dataset_oz = "~/sleepedf/EDF_20/pz_oz"
dataset_eog = "~/sleepedf/EDF_20/eog"  
cz_files = glob.glob(os.path.join(dataset_cz, "*.npz"))
oz_files = glob.glob(os.path.join(dataset_oz, "*.npz"))
eog_files = glob.glob(os.path.join(dataset_eog, "*.npz"))  
seq_sids = load_seq_ids("~/sleepedf/EDF_20/sleepedf.txt")

def main(fold_idx, cz_files, oz_files, eog_files, seq_sids):

    def make_dataset(sids):
        d = {k: load_and_process_data(ds, f, sids)
             for k, ds, f in [("cz",  dataset_cz,  cz_files),
                               ("oz",  dataset_oz,  oz_files),
                               ("eog", dataset_eog, eog_files)]}
        return create_multi_tensor_dataset(
            d["cz"][0] + d["oz"][0], d["cz"][0] - d["oz"][0],
            d["eog"][0], d["cz"][1], device)

    test_sids  = np.array_split(seq_sids, 20)[fold_idx]
    train_sids = np.setdiff1d(seq_sids, test_sids)
    
    # 读取 GAN 样本，仅用于校准集
    gan_data = np.load(f"diffu_data/diffusion{fold_idx}.npz")
    calib_dataset = create_multi_tensor_dataset(
    gan_data["X"][:, 0, :] + gan_data["X"][:, 1, :],  # cz + oz
    gan_data["X"][:, 0, :] - gan_data["X"][:, 1, :],  # cz - oz
    gan_data["X"][:, 2, :],
    gan_data["y"], device)
    N_CALIB=len(calib_dataset)
    
    # 训练集 = 全部 train_sids
    train_all = make_dataset(train_sids)
    train_size = int(0.9 * len(train_all))
    train_dataset, val_dataset = random_split(train_all, [train_size, len(train_all) - train_size])
    
    # 数据增强（10% 训练样本）
    augmented_samples = [
        (augment_N1(train_dataset[idx][0]), train_dataset[idx][1])
        for idx in random.sample(range(len(train_dataset)), int(0.1 * len(train_dataset)))
    ]
    augmented_train_dataset = CombinedDataset(
        [train_dataset[i] for i in range(len(train_dataset))] + augmented_samples)
    
    test_dataset = make_dataset(test_sids)
    print(f"Fold {fold_idx} | train={len(augmented_train_dataset)} val={len(val_dataset)} "
          f"Calib={len(calib_dataset)} test={len(test_dataset)}")
    
    train_loader = DataLoader(augmented_train_dataset, batch_size=32,     shuffle=True)
    valid_loader = DataLoader(val_dataset,             batch_size=32,     shuffle=False)
    calib_loader = DataLoader(calib_dataset,           batch_size=N_CALIB, shuffle=False)
    test_loader  = DataLoader(test_dataset,            batch_size=32,     shuffle=False)

    model     = SleepStageModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    models = train(model, train_loader, valid_loader, criterion, optimizer, epochs=50)
    return models, calib_loader, test_loader


ALPHAS = [0.5, 0.4, 0.3, 0.2]
all_results = {a: {"preds": [], "labels": []} for a in ALPHAS}

for fold_idx in range(20):
    models, calib_loader, test_loader = main(fold_idx, cz_files, oz_files, eog_files, seq_sids)
    for alpha in ALPHAS:
        thresholds = calibrate_threshold_ensemble(models, calib_loader, alpha=alpha)
        preds, labels = np_predict_ensemble(models, test_loader, thresholds)
        all_results[alpha]["preds"].append(preds)
        all_results[alpha]["labels"].append(labels)

for alpha in ALPHAS:
    all_preds_cat  = np.concatenate(all_results[alpha]["preds"])
    all_labels_cat = np.concatenate(all_results[alpha]["labels"])
    cm         = confusion_matrix(all_labels_cat, all_preds_cat)
    per_class  = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    print(f"\n===== Alpha={alpha} =====")
    print(f"Overall Accuracy : {accuracy_score(all_labels_cat, all_preds_cat)*100:.2f}%")
    print(f"Macro F1         : {f1_score(all_labels_cat, all_preds_cat, average='macro'):.4f}")
    print(f"Cohen's Kappa    : {cohen_kappa_score(all_labels_cat, all_preds_cat):.4f}")
    header = f"{'':>5}" + "".join(f"{c:>7}" for c in CLASS_NAMES)
    print(f"Confusion matrix:\n{header}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>3} " + "".join(f"{v:>7}" for v in row))
    print("Per-class accuracy: " + "  ".join(
            f"{CLASS_NAMES[i]}={per_class[i]*100:.1f}%" for i in range(len(CLASS_NAMES))))