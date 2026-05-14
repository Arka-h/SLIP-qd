"""
Train a linear classifier on frozen SLIP visual features over the QuickDraw
holdout classes, then evaluate on the holdout test split.

Outputs a table comparing zero-shot vs linear-probe accuracy.

Usage:
    python linear_probe.py \
        --checkpoint output/slip_vitb32_qd14_lora_ep_10_hpset_7vpoqhty_set_1/checkpoint_best.pt \
        --root /path/to/quickdraw \
        --model SLIP_VITB32 \
        --lora-rank 16 --lora-alpha 16 \
        --gpu 0
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets
import models
import utils
from datasets import QD_Val, QD_HOLDOUT_CLASSES
from tokenizer import SimpleTokenizer
from peft import LoraConfig, get_peft_model


# ── args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root',       required=True)
    p.add_argument('--model',      default='SLIP_VITB32')
    p.add_argument('--gpu',        default=0, type=int)
    p.add_argument('--workers',    default=4, type=int)
    p.add_argument('--batch-size', default=256, type=int)
    # LoRA must match what was used during training
    p.add_argument('--use-lora',   action='store_true', default=True)
    p.add_argument('--lora-rank',  default=16, type=int)
    p.add_argument('--lora-alpha', default=16, type=float)
    p.add_argument('--lora-dropout', default=0.0, type=float)
    # Linear probe hyper-params
    p.add_argument('--epochs',     default=50, type=int)
    p.add_argument('--lr',         default=0.1, type=float)
    p.add_argument('--wd',         default=0.0, type=float)
    # SSL model shape (must match checkpoint)
    p.add_argument('--ssl-mlp-dim', default=4096, type=int)
    p.add_argument('--ssl-emb-dim', default=256,  type=int)
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(args):
    model = getattr(models, args.model)(
        ssl_mlp_dim=args.ssl_mlp_dim,
        ssl_emb_dim=args.ssl_emb_dim,
    )
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model.visual = get_peft_model(model.visual, lora_config)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    # strip DDP prefix if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model = model.cuda(args.gpu)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, gpu):
    feats, labels = [], []
    for images, targets in loader:
        images = images.cuda(gpu, non_blocking=True)
        f = utils.get_model(model).encode_image(images)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


# ── zero-shot eval ─────────────────────────────────────────────────────────────

@torch.no_grad()
def zero_shot_accuracy(model, loader, class_names, gpu):
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'templates.json')) as f:
        templates = json.load(f).get('quickdraw', datasets.QD_TEMPLATES)

    tokenizer = SimpleTokenizer()
    text_features = []
    for name in class_names:
        texts = tokenizer([t.format(name) for t in templates]).cuda(gpu)
        emb = utils.get_model(model).encode_text(texts)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(0)
        emb = emb / emb.norm()
        text_features.append(emb)
    text_features = torch.stack(text_features)  # (C, D)

    correct1 = correct5 = total = 0
    for images, targets in loader:
        images  = images.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        img_f = utils.get_model(model).encode_image(images)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = img_f @ text_features.t()
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        correct1 += acc1.item() * n / 100
        correct5 += acc5.item() * n / 100
        total    += n

    return 100 * correct1 / total, 100 * correct5 / total


# ── linear probe training ─────────────────────────────────────────────────────

def train_linear_probe(train_feats, train_labels, num_classes, feat_dim, args):
    head = nn.Linear(feat_dim, num_classes).cuda(args.gpu)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    optimizer = torch.optim.SGD(head.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    X = train_feats.cuda(args.gpu)
    y = train_labels.long().cuda(args.gpu)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"  [probe epoch {epoch+1:3d}/{args.epochs}] loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.4f}")

    head.eval()
    return head


# ── linear probe evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def eval_linear_probe(head, test_feats, test_labels, gpu):
    X = test_feats.cuda(gpu)
    y = test_labels.cuda(gpu)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=512)

    correct1 = correct5 = total = 0
    for xb, yb in loader:
        logits = head(xb)
        a1, a5 = accuracy(logits, yb, topk=(1, 5))
        n = xb.size(0)
        correct1 += a1.item() * n / 100
        correct5 += a5.item() * n / 100
        total    += n

    return 100 * correct1 / total, 100 * correct5 / total


# ── utils ─────────────────────────────────────────────────────────────────────

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def make_loader(root, split, transform, batch_size, workers):
    ds = QD_Val(root=root, holdout_classes=QD_HOLDOUT_CLASSES,
                split=split, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) \
        if 'VITB32' in args.model else \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print("=> loading model")
    model = load_model(args)

    train_loader = make_loader(args.root, 'train', val_transform, args.batch_size, args.workers)
    test_loader  = make_loader(args.root, 'test',  val_transform, args.batch_size, args.workers)

    class_names = QD_HOLDOUT_CLASSES
    num_classes = len(class_names)

    # ── zero-shot baseline ────────────────────────────────────────────────────
    print("\n=> zero-shot evaluation on test split")
    zs_top1, zs_top5 = zero_shot_accuracy(model, test_loader, class_names, args.gpu)
    print(f"   Zero-shot  Acc@1={zs_top1:.2f}%  Acc@5={zs_top5:.2f}%")

    # ── extract features ──────────────────────────────────────────────────────
    print("\n=> extracting train features")
    train_feats, train_labels = extract_features(model, train_loader, args.gpu)
    print(f"   train: {train_feats.shape}  labels: {train_labels.shape}")

    print("=> extracting test features")
    test_feats, test_labels = extract_features(model, test_loader, args.gpu)
    print(f"   test:  {test_feats.shape}   labels: {test_labels.shape}")

    feat_dim = train_feats.shape[1]

    # ── train linear probe ────────────────────────────────────────────────────
    print(f"\n=> training linear probe ({feat_dim}d → {num_classes} classes, {args.epochs} epochs)")
    head = train_linear_probe(train_feats, train_labels, num_classes, feat_dim, args)

    # ── evaluate linear probe ─────────────────────────────────────────────────
    print("\n=> evaluating linear probe on test split")
    lp_top1, lp_top5 = eval_linear_probe(head, test_feats, test_labels, args.gpu)

    # ── summary ───────────────────────────────────────────────────────────────
    random_top1 = 100.0 / num_classes
    random_top5 = 100.0 * min(5, num_classes) / num_classes

    print("\n" + "=" * 52)
    print(f"  {'':20s}  {'Acc@1':>8s}  {'Acc@5':>8s}")
    print(f"  {'Random baseline':20s}  {random_top1:>8.2f}%  {random_top5:>8.2f}%")
    print(f"  {'Zero-shot (CLIP)':20s}  {zs_top1:>8.2f}%  {zs_top5:>8.2f}%")
    print(f"  {'Linear probe':20s}  {lp_top1:>8.2f}%  {lp_top5:>8.2f}%")
    print("=" * 52)

    gap = lp_top1 - zs_top1
    print(f"\n  Linear probe - Zero-shot gap: {gap:+.2f}pp")
    if gap > 20:
        print("  => Large gap: encoder features are good, text-image alignment is the bottleneck.")
    elif gap > 5:
        print("  => Moderate gap: both encoder and alignment are partially working.")
    else:
        print("  => Small gap: encoder features and text alignment are consistent.")


if __name__ == '__main__':
    main()
