"""训练广告分类器。数据是一个 CSV，两列：text（邮件内容）、label（1=广告，0=正常）。"""
import argparse
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import config
from tokenizer import Tokenizer
from model import AdClassifier


def load_csv(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def make_loader(tok, texts, labels, batch, shuffle):
    encoded = [tok.encode(t, config.MAX_LEN) for t in texts]
    ids = torch.tensor([e[0] for e in encoded])
    masks = torch.tensor([e[1] for e in encoded])
    return DataLoader(TensorDataset(ids, masks, torch.tensor(labels)), batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = correct = total = 0
    for x, m, y in loader:
        pred = model(x.to(device), m.to(device)).argmax(-1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return correct / total, prec, rec, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts, labels = load_csv(args.data)
    idx = list(range(len(texts)))
    random.shuffle(idx)
    split = int(len(idx) * 0.9)
    tr, va = idx[:split], idx[split:]
    tr_texts, tr_labels = [texts[i] for i in tr], [labels[i] for i in tr]
    va_texts, va_labels = [texts[i] for i in va], [labels[i] for i in va]

    n_ad = sum(tr_labels)
    n_normal = len(tr_labels) - n_ad
    if n_ad == 0 or n_normal == 0:
        raise SystemExit("训练数据里需要同时有广告和正常邮件。")
    print(f"训练集 {len(tr)} 封（广告 {n_ad}，正常 {n_normal}），验证集 {len(va)} 封，设备：{device}")

    tok = Tokenizer.build(tr_texts)
    train_loader = make_loader(tok, tr_texts, tr_labels, args.batch, True)
    val_loader = make_loader(tok, va_texts, va_labels, args.batch, False)

    model_cfg = dict(vocab_size=len(tok.vocab), max_len=config.MAX_LEN,
                     d_model=128, n_heads=4, n_layers=2, dropout=0.1)
    model = AdClassifier(**model_cfg).to(device)
    print(f"词表大小 {len(tok.vocab)}，模型参数 {sum(p.numel() for p in model.parameters()):,}")

    # 广告和正常邮件数量往往不平衡，按比例给少数类更大的权重
    weights = torch.tensor([len(tr_labels) / (2 * n_normal), len(tr_labels) / (2 * n_ad)], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            loss = loss_fn(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(y)
        acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"第 {epoch} 轮  loss {total_loss / len(tr):.4f}  准确率 {acc:.3f}  "
              f"精确率 {prec:.3f}  召回率 {rec:.3f}  F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_cfg": model_cfg, "state_dict": model.state_dict(), "vocab": tok.vocab},
                       config.MODEL_PATH)
    print(f"训练完成，最佳 F1 {best_f1:.3f}，模型已保存到 {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
