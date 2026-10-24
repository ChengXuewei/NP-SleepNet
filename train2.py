import os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset, random_split, ConcatDataset
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, accuracy_score
from EEG_data import load_and_process_data, create_multi_tensor_dataset
from utils_eeg import load_seq_ids
from logger import get_logger
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
logger = get_logger("train_out5", level="info")


class CombinedDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

dataset_cz  = "~/sleepedf/EDF_20/fpz_cz"
dataset_oz  = "~/sleepedf/EDF_20/pz_oz"
dataset_eog = "~/sleepedf/EDF_20/eog"
cz_files  = glob.glob(os.path.join(dataset_cz,  "*.npz"))
oz_files  = glob.glob(os.path.join(dataset_oz,  "*.npz"))
eog_files = glob.glob(os.path.join(dataset_eog, "*.npz"))
seq_sids  = load_seq_ids("~/sleepedf/EDF_20/sleepedf.txt")

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

    # 每折用固定 rng 选一个校准 subject
    rng = np.random.default_rng(seed=42+fold_idx)
    calib_sid   = rng.choice(train_sids, size=1, replace=False)
    remain_sids = np.setdiff1d(train_sids, calib_sid)

    # 校准集：仅保留 N1；非 N1 并回训练集
    calib_full   = make_dataset(calib_sid)
    calib_n1_idx = [i for i in range(len(calib_full)) if calib_full[i][1].item() == N1_LABEL]
    calib_non_n1 = [calib_full[i] for i in range(len(calib_full)) if i not in set(calib_n1_idx)]
    calib_dataset = CombinedDataset([calib_full[i] for i in calib_n1_idx])
    N_CALIB = len(calib_n1_idx)

    # 训练集 = remain + 校准 subject 的非 N1 样本
    train_base    = make_dataset(remain_sids)
    train_all     = CombinedDataset([train_base[i] for i in range(len(train_base))] + calib_non_n1)
    test_dataset  = make_dataset(test_sids)

    # 训练集内划分 90/10 验证集（供 ensemble checkpoint 筛选）
    train_size = int(0.9 * len(train_all))
    val_size   = len(train_all) - train_size
    train_dataset, val_dataset = random_split(train_all, [train_size, val_size])

    # 计算需要增强的样本数量 (10% of train_dataset)
    augment_size = int(0.1 * len(train_dataset))
    indices_to_augment = random.sample(range(len(train_dataset)), augment_size)

    # 对选中的样本进行数据增强
    augmented_samples = []
    for idx in indices_to_augment:
        original_data, original_label = train_dataset[idx]
        augmented_data = augment_N1(original_data)  # 假设augment_N1接受单个样本并返回增强后的数据
        augmented_samples.append((augmented_data, original_label))

    # 首先获取原始训练集的所有样本
    original_samples = [train_dataset[i] for i in range(len(train_dataset))]
    # 然后合并原始样本和增强样本
    combined_samples = original_samples + augmented_samples

    # 创建新的训练集
    augmented_train_dataset = CombinedDataset(combined_samples)
    print(f"Augmented_sample size: {len(augmented_train_dataset)}")

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

    logger.info(f"\n===== Alpha={alpha} =====")
    logger.info(f"Overall Accuracy : {accuracy_score(all_labels_cat, all_preds_cat)*100:.2f}%")
    logger.info(f"Macro F1         : {f1_score(all_labels_cat, all_preds_cat, average='macro'):.4f}")
    logger.info(f"Cohen's Kappa    : {cohen_kappa_score(all_labels_cat, all_preds_cat):.4f}")
    header = f"{'':>5}" + "".join(f"{c:>7}" for c in CLASS_NAMES)
    logger.info(f"Confusion matrix:\n{header}")
    for i, row in enumerate(cm):
        logger.info(f"  {CLASS_NAMES[i]:>3} " + "".join(f"{v:>7}" for v in row))
    logger.info("Per-class accuracy: " + "  ".join(
            f"{CLASS_NAMES[i]}={per_class[i]*100:.1f}%" for i in range(len(CLASS_NAMES))))