"""Fine-tune MobileNetV2. Expects data/<denomination>/*.jpg. Run: python train.py"""
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms as T
from torchvision.datasets import ImageFolder

from common import MEAN, SIZE, STD, load

DATA = "data"
EPOCHS = 15


def splits(train_tf=None, eval_tf=None, seed=0):
    """Deterministic 70/15/15 split -> (classes, train, val, test). export.py reuses this."""
    ds = lambda tf: ImageFolder(DATA, tf, loader=load)
    full = ds(None)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(len(full), generator=g).tolist()
    a, b = int(0.7 * len(ids)), int(0.85 * len(ids))
    return (full.classes, Subset(ds(train_tf), ids[:a]),
            Subset(ds(eval_tf), ids[a:b]), Subset(ds(eval_tf), ids[b:]))


@torch.no_grad()
def accuracy(model, ds):
    model.eval()
    ok = sum((model(x).argmax(1) == y).sum().item() for x, y in DataLoader(ds, 64, num_workers=4))
    return ok / len(ds)


def main():
    norm = [T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())]
    # No flips (mirrored text isn't a real banknote) and no crops (could cut off the digits).
    aug = T.Compose([T.Resize((SIZE, SIZE)), T.RandomAffine(15, (0.05, 0.05), (0.9, 1.1)),
                     T.ColorJitter(0.3, 0.3, 0.3, 0.02), *norm])
    ev = T.Compose([T.Resize((SIZE, SIZE)), *norm])
    classes, tr, va, te = splits(aug, ev)
    json.dump(classes, open("labels.json", "w"))  # ImageFolder sorts as strings: 100,1000,20,50,500
    print("classes:", classes, "| train/val/test:", len(tr), len(va), len(te))

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, len(classes))
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    dl = DataLoader(tr, 32, shuffle=True, num_workers=4)

    best = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        acc = accuracy(model, va)
        print(f"epoch {ep + 1:2d}  val_acc {acc:.3f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), "model.pt")

    model.load_state_dict(torch.load("model.pt"))
    print(f"TEST accuracy (PyTorch): {accuracy(model, te):.3f}")


if __name__ == "__main__":
    main()
