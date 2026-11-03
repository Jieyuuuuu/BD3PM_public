import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from bd3pm.data import PianoRollDataset
from bd3pm.diffusion import BinaryDiffusion
from bd3pm.models import BD3PM


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[n]

    def state_dict(self, model):
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        sd.update(self.shadow)
        return sd


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith("bias") or "norm" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.npz")
    parser.add_argument("--val", default="data/val.npz")
    parser.add_argument("--out", default="checkpoints/bd3pm")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    loader = DataLoader(PianoRollDataset(args.train, augment=True), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(PianoRollDataset(args.val), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    model = BD3PM()
    diffusion = BinaryDiffusion(model).to(device)
    bce = [nn.BCEWithLogitsLoss(pos_weight=torch.full((88,), w, device=device)) for w in (2.0, 1.5)]
    optimizer = optim.AdamW(param_groups(model, 0.01), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, 0.9995)
    history, best, start = {"train": [], "val": []}, float("inf"), 0

    def compute_loss(x0):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = diffusion(x0)
            return 1.5 * bce[0](logits[:, :, 0::2], x0[:, :, 0::2]) + bce[1](logits[:, :, 1::2], x0[:, :, 1::2])

    state_path = os.path.join(args.out, "train_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        ema.shadow = st["ema"]
        history, best, start = st["history"], st["best"], st["epoch"] + 1
        torch.set_rng_state(st["rng_cpu"].cpu())
        torch.cuda.set_rng_state(st["rng_cuda"].cpu())

    for epoch in range(start, args.epochs):
        model.train()
        total = 0.0
        for x0 in tqdm(loader, desc=f"epoch {epoch + 1}"):
            loss = compute_loss(x0.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scale <= scaler.get_scale():
                ema.update(model)
            total += loss.item()
        history["train"].append(total / len(loader))

        model.eval()
        with torch.no_grad():
            val = sum(compute_loss(x0.to(device)).item() for x0 in val_loader)
        history["val"].append(val / len(val_loader))
        scheduler.step(history["val"][-1])
        print(f"epoch {epoch + 1}: train {history['train'][-1]:.4f} val {history['val'][-1]:.4f}")

        if history["val"][-1] < best:
            best = history["val"][-1]
            torch.save(ema.state_dict(model), os.path.join(args.out, "best_ema.pt"))

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "ema": ema.shadow,
                    "history": history, "best": best, "epoch": epoch,
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()},
                   state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    json.dump(history, open(os.path.join(args.out, "loss_history.json"), "w"), indent=1)
    os.remove(state_path)


if __name__ == "__main__":
    main()
