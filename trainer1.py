import torch
import numpy as np
import copy
from model4 import SleepStageModel

N1_LABEL = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ── 训练，返回 top-k checkpoint 模型列表 ──────────────────────────────────────
def train(model, train_loader, valid_loader, criterion, optimizer, epochs=50, k=10):
    best_models = {}
    best_val_acc = np.zeros(k)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        val_acc = _accuracy(model, valid_loader)
        if val_acc > best_val_acc.min():
            j = best_val_acc.argmin()
            best_val_acc[j] = val_acc
            best_models[j] = copy.deepcopy(model.state_dict())
            train_acc = _accuracy(model, train_loader)
            if epoch==0 or (epoch+1) % 10 ==0:
                print(f"Epoch {epoch+1:02d} | update slot {j+1} | "
                      f"train={train_acc*100:.2f}%  val={val_acc*100:.2f}%")

    models = []
    for state_dict in best_models.values():
        m = SleepStageModel().to(device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
    return models


# ── 辅助：计算准确率 ───────────────────────────────────────────────────────────
@torch.no_grad()
def _accuracy(model, loader):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        correct += (model(inputs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── 每个模型独立校准阈值 ───────────────────────────────────────────────────────
@torch.no_grad()
def calibrate_threshold_ensemble(models, calib_loader, alpha=0.2):
    """返回每个模型对应的 NP 阈值列表"""
    thresholds = []
    for m in models:
        scores = []
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            prob = torch.softmax(m(x), dim=1)[:, N1_LABEL]
            scores.append(prob[y == N1_LABEL].cpu())
        scores = np.sort(torch.cat(scores).numpy())
        t = float(scores[int(np.floor(len(scores) * alpha))])
        thresholds.append(t)
    print(f"Thresholds: {[f'{t:.4f}' for t in thresholds]}")
    return thresholds


@torch.no_grad()
def np_predict_ensemble(models, loader, thresholds=None):
    """各模型预测后多数投票。thresholds=None 时用普通 softmax argmax。"""
    all_model_preds, all_labels = [], None

    for i, m in enumerate(models):
        threshold = thresholds[i] if thresholds is not None else None
        preds, labels = _np_predict_single(m, loader, threshold)
        all_model_preds.append(preds)
        all_labels = labels

    votes = np.stack(all_model_preds, axis=0)
    final = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=5).argmax(), 0, votes
    )
    return final, all_labels


@torch.no_grad()
def _np_predict_single(model, loader, threshold=None):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1)
        if threshold is not None:
            prob_non_n1 = prob.clone()
            prob_non_n1[:, N1_LABEL] = -1
            pred = torch.where(
                prob[:, N1_LABEL] >= threshold,
                torch.full((x.size(0),), N1_LABEL, device=x.device),
                prob_non_n1.argmax(1)
            )
        else:
            pred = prob.argmax(1)
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()