import torch
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        if epoch==0 or (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Train Acc: {correct/total*100:.2f}%")

    return model

@torch.no_grad()
def calibrate_threshold(model, calib_loader, alpha=0.2, n1_label=1):
    model.eval()
    scores = []
    for x, y in calib_loader:
        x, y = x.to(device), y.to(device)
        prob = torch.softmax(model(x), dim=1)[:, n1_label]
        scores.append(prob[y == n1_label].cpu())
    scores = np.sort(torch.cat(scores).numpy())
    threshold = scores[int(np.floor(len(scores) * alpha))]
    print(f"NP threshold: {threshold:.4f}  (n={len(scores)}, alpha={alpha})")
    return threshold

@torch.no_grad()
def np_predict(model, loader, threshold=None, n1_label=1):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1)
        if threshold is None:
            pred = prob.argmax(1)
        else:
            prob_non_n1 = prob.clone()
            prob_non_n1[:, n1_label] = -1
            pred = torch.where(
                prob[:, n1_label] >= threshold,
                torch.full_like(prob_non_n1.argmax(1), n1_label),
                prob_non_n1.argmax(1)
            )
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()