"""The symbol reader: a small convolutional network (about 150,000 weights) that looks at one
28x28 glyph and says how likely each of the 22 symbols is.

Training mixes sources per class (synthetic, MNIST, HASYv2, your own writing), augments
every batch (slant, rotation, size, elastic wobble, pen width, noise), and ends with
temperature scaling: one number that makes the probabilities honest, so "60% a 1"
really is right about 60% of the time. Honest probabilities are what let context
decide the close calls well.
"""
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .glyphs import CLASSES, PAIRS

K = len(CLASSES)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Net(nn.Module):
    def __init__(self, k=K, w=24):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(c(1, w), c(w, w), nn.MaxPool2d(2), c(w, 2 * w), c(2 * w, 2 * w), nn.MaxPool2d(2),
                               c(2 * w, 64), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, k))

    def forward(self, x):
        return self.head(self.f(x))


# ------------------------------------------------------------------ batch augmentation on the device
def augment_batch(x, gen, strength=1.0):
    """x: (B,1,28,28) in [0,1]. Random affine + elastic + pen width + noise, all in one pass."""
    B, dev = x.shape[0], x.device
    r = lambda *s: torch.randn(*s, generator=gen).to(dev)
    u = lambda *s: torch.rand(*s, generator=gen).to(dev)
    ang = r(B) * (8 * math.pi / 180) * strength
    shear = r(B) * 0.15 * strength
    sc = 1 + r(B) * 0.08 * strength
    asp = 1 + r(B) * 0.08 * strength
    tx, ty = r(B) * 0.06 * strength, r(B) * 0.06 * strength
    cos, sin = torch.cos(ang), torch.sin(ang)
    a11 = cos * sc * asp
    a12 = (-sin + cos * shear) * sc / asp
    a21 = sin * sc * asp
    a22 = (cos + sin * shear) * sc / asp
    theta = torch.stack([torch.stack([a11, a12, tx], 1), torch.stack([a21, a22, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    if strength > 0:                                                   # elastic wobble on half the batch
        d = F.interpolate(r(B, 2, 4, 4) * 0.035 * strength, size=(28, 28), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        grid = grid + d * (u(B, 1, 1, 1) < 0.5).float()
    y = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    pick = u(B, 1, 1, 1)
    dil = F.max_pool2d(y, 3, 1, 1)
    ero = -F.max_pool2d(-y, (1, 2), 1, (0, 0))
    ero = F.pad(ero, (0, 1, 0, 0))
    y = torch.where(pick < 0.18, 0.5 * (y + dil), torch.where(pick < 0.3, ero, y))
    y = y + r(*y.shape) * 0.04 * (u(B, 1, 1, 1) < 0.3).float()
    return y.clamp(0, 1)


# ------------------------------------------------------------------ training
def train(X, y, Xval=None, yval=None, steps=1500, batch=128, lr=2e-3, augment=True, init=None, seed=0,
          progress=lambda f, m, live=None: None, should_stop=lambda: False):
    """Returns (net, temperature, curve)."""
    dev = device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = Net().to(dev)
    if init is not None:
        net.load_state_dict(init)
    Xt = torch.tensor(X).unsqueeze(1)
    yt = torch.tensor(y)
    n = len(yt)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    curve = []
    net.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        if should_stop():
            raise InterruptedError("Stopped.")
        idx = torch.randint(0, n, (min(batch, n),), generator=gen)
        xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
        if augment:
            xb = augment_batch(xb, gen)
        loss = F.cross_entropy(net(xb), yb, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 30) == 0 or step == steps:
            row = {"step": step, "loss": loss.item()}
            if Xval is not None and step % max(1, steps // 6) == 0:
                row["val_acc"] = accuracy(net, Xval, yval)
                net.train()
            curve.append(row)
            progress(step / steps, f"Step {step} of {steps}: loss {row['loss']:.3f}" + (f", held-out accuracy {row['val_acc'] * 100:.1f}%" if "val_acc" in row else ""),
                     {"curve": curve, "seconds": time.time() - t0})
    net.eval()
    T = fit_temperature(net, Xval, yval) if Xval is not None and len(yval) >= 50 else 1.0
    return net, T, curve


@torch.no_grad()
def logits(net, X, bs=1024):
    dev = next(net.parameters()).device
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(net(torch.tensor(X[i:i + bs]).unsqueeze(1).to(dev)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, K)


def accuracy(net, X, y):
    if X is None or len(y) == 0:
        return None
    return float((logits(net, X).argmax(1).numpy() == y).mean())


def fit_temperature(net, X, y):
    L = logits(net, X)
    t = torch.tensor(y)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / logT.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.detach().exp().clamp(0.3, 5.0).item())


def evaluate(net, T, X, y):
    """Accuracy, per-class accuracy, confusion, the confusable pairs, and calibration (ECE)."""
    if X is None or len(y) == 0:
        return None
    P = F.softmax(logits(net, X) / T, 1).numpy()
    pred = P.argmax(1)
    conf = np.zeros((K, K), np.int64)
    np.add.at(conf, (y, pred), 1)
    per = {}
    for k in range(K):
        n = conf[k].sum()
        if n:
            per[CLASSES[k]] = {"n": int(n), "acc": float(conf[k, k] / n)}
    pairs = {}
    for a, b in PAIRS:
        ia, ib = CLASSES.index(a), CLASSES.index(b)
        na, nb = conf[ia].sum(), conf[ib].sum()
        if na or nb:
            pairs[f"{a}|{b}"] = {"a_as_b": float(conf[ia, ib] / na) if na else None, "b_as_a": float(conf[ib, ia] / nb) if nb else None,
                                 "n": int(na + nb)}
    top = P.max(1)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (top > lo) & (top <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - top[m].mean())
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "per_class": per, "pairs": pairs, "ece": float(ece),
            "confusion": conf.tolist()}


# ------------------------------------------------------------------ saved models
class Registry:
    def __init__(self, data_dir: Path, starter: Path = None):
        self.dir = Path(data_dir) / "models"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.settings = Path(data_dir) / "settings.json"
        self._loaded = {}
        if starter and starter.exists() and not any(self.dir.glob("*/meta.json")):
            d = self.dir / "starter"
            d.mkdir(exist_ok=True)
            (d / "model.pt").write_bytes(starter.read_bytes())
            meta = json.loads(starter.with_suffix(".json").read_text())
            (d / "meta.json").write_text(json.dumps(meta))
            self.set_active("starter")

    def save(self, net, T, meta):
        mid = meta.get("id") or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        d = self.dir / mid
        d.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, d / "model.pt")
        meta = {**meta, "id": mid, "temperature": T, "classes": CLASSES, "created": meta.get("created", time.time())}
        (d / "meta.json").write_text(json.dumps(meta))
        return meta

    def list(self):
        out = []
        for f in sorted(self.dir.glob("*/meta.json"), key=lambda p: -p.stat().st_mtime):
            try:
                m = json.loads(f.read_text())
            except Exception:
                continue
            m.pop("curve", None)
            for ev in (m.get("eval") or {}).values():
                if ev:
                    ev.pop("confusion", None)
            out.append(m)
        return out

    def meta(self, mid):
        return json.loads((self.dir / mid / "meta.json").read_text())

    def active_id(self):
        try:
            return json.loads(self.settings.read_text()).get("active_model")
        except Exception:
            return None

    def set_active(self, mid):
        if not (self.dir / mid / "meta.json").exists():
            raise ValueError(f"No model {mid}.")
        try:
            s = json.loads(self.settings.read_text())
        except Exception:
            s = {}
        s["active_model"] = mid
        self.settings.write_text(json.dumps(s))

    def load(self, mid=None):
        mid = mid or self.active_id()
        if not mid:
            return None, None, None
        if mid not in self._loaded:
            meta = self.meta(mid)
            net = Net()
            net.load_state_dict(torch.load(self.dir / mid / "model.pt", map_location="cpu", weights_only=True))
            net.eval()
            self._loaded = {mid: (net, float(meta.get("temperature", 1.0)), meta)}
        return self._loaded[mid]

    def delete(self, mid):
        import shutil

        if mid == self.active_id():
            raise ValueError("That model is in use; choose another first.")
        shutil.rmtree(self.dir / mid, ignore_errors=True)
        self._loaded.pop(mid, None)


@torch.no_grad()
def predict(net, T, glyphs):
    """glyphs: (N,28,28) -> calibrated probabilities (N,K)."""
    if len(glyphs) == 0:
        return np.zeros((0, K), np.float32)
    return F.softmax(logits(net, np.asarray(glyphs, np.float32)) / T, 1).numpy()
