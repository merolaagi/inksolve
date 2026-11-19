"""Run trace_code.training_step under sys.settrace and record, line by line, what Python
Tutor would show: the line just run and the next one, every frame's variables, and every
tensor they point to, including the model's own weights and gradients.

Tensors are recorded once per version (PyTorch counts in-place changes in t._version), so a
weight appears twice: before and after its update, with the difference kept too."""
import base64
import inspect
import sys
import types

import numpy as np
import torch

from . import trace_code as TC
from .glyphs import CLASSES

TRACED = {"forward", "training_step"}
SRC_FILE = inspect.getsourcefile(TC)


def wrap(model):
    f, hd = model.f, model.head
    mods = {"conv1": f[0][0], "bn1": f[0][1], "conv2": f[1][0], "bn2": f[1][1], "conv3": f[3][0], "bn3": f[3][1],
            "conv4": f[4][0], "bn4": f[4][1], "conv5": f[6][0], "bn5": f[6][1], "fc1": hd[2], "fc2": hd[5]}
    ns = types.SimpleNamespace(**mods)
    ns.layers = list(mods.values())
    ns.parameters = model.parameters
    ns.zero_grad = lambda: model.zero_grad(set_to_none=True)
    names = {id(m): n for n, m in mods.items()}
    return ns, mods, names


def _layout(a):
    """Any tensor -> a 2D picture: conv weights as a grid of 3x3 tiles, feature maps as a grid of maps."""
    a = np.asarray(a, np.float32)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim == 2:
        return a
    if a.ndim == 4 and a.shape[0] == 1:                   # (1, C, H, W) activations
        a = a[0]
        C, H, W = a.shape
        cols = int(np.ceil(np.sqrt(C)))
        rows = int(np.ceil(C / cols))
        out = np.full((rows * (H + 1) - 1, cols * (W + 1) - 1), np.nan, np.float32)
        for c in range(C):
            r, k = divmod(c, cols)
            out[r * (H + 1):r * (H + 1) + H, k * (W + 1):k * (W + 1) + W] = a[c]
        return out
    if a.ndim == 4 and a.shape[1] == 1:                   # (O, 1, kh, kw): a grid of O small kernels
        O, _, kh, kw = a.shape
        cols = int(np.ceil(np.sqrt(O)))
        rows = int(np.ceil(O / cols))
        out = np.full((rows * (kh + 1) - 1, cols * (kw + 1) - 1), np.nan, np.float32)
        for o in range(O):
            r, k = divmod(o, cols)
            out[r * (kh + 1):r * (kh + 1) + kh, k * (kw + 1):k * (kw + 1) + kw] = a[o, 0]
        return out
    if a.ndim == 4:                                       # (O, I, kh, kw) weights
        O, I, kh, kw = a.shape
        out = np.full((O * (kh + 1) - 1, I * (kw + 1) - 1), np.nan, np.float32)
        for o in range(O):
            for i in range(I):
                out[o * (kh + 1):o * (kh + 1) + kh, i * (kw + 1):i * (kw + 1) + kw] = a[o, i]
        return out
    return a.reshape(a.shape[0], -1)


def _enc(img):
    """Signed values -> bytes: 0 = empty, 1..255 with 128 = zero."""
    m = float(np.nanmax(np.abs(img))) if np.isfinite(img).any() else 0.0
    q = np.zeros(img.shape, np.uint8)
    ok = np.isfinite(img)
    if m > 0:
        q[ok] = np.clip(np.round(img[ok] / m * 127) + 128, 1, 255)
    else:
        q[ok] = 128
    return base64.b64encode(q.tobytes()).decode(), m


def _preview(a, n=8):
    f = np.asarray(a, np.float32).reshape(-1)
    return [float(x) for x in f[:n]]


class Recorder:
    def __init__(self, model, mods, names):
        self.model, self.mods, self.names = model, mods, names
        self.objects = {}           # oid -> description
        self.keys = {}              # (id, version) -> oid
        self.last_by_id = {}        # id(tensor) -> (oid, numpy copy)
        self.param_name = {}
        for n, m in mods.items():
            for pn in ("weight", "bias", "running_mean", "running_var"):
                t = getattr(m, pn, None)
                if isinstance(t, torch.Tensor):
                    self.param_name[id(t)] = f"net.{n}.{pn}"
        self.steps = []
        self.prev_params = {}

    def obj(self, t, label=None):
        key = (id(t), t._version)
        if key in self.keys:
            return self.keys[key]
        oid = f"t{len(self.objects)}"
        a = t.detach().cpu().numpy().copy()
        img = _layout(a)
        heat, m = _enc(img)
        d = {"id": oid, "label": label or self.param_name.get(id(t)), "shape": list(a.shape), "n": int(a.size),
             "dtype": str(t.dtype).replace("torch.", ""), "heat": heat, "hw": list(img.shape), "max": m,
             "stats": {"min": float(a.min()) if a.size else 0, "max": float(a.max()) if a.size else 0,
                       "mean": float(a.mean()) if a.size else 0, "abs": float(np.abs(a).mean()) if a.size else 0,
                       "zeros": float((a == 0).mean()) if a.size else 0},
             "preview": _preview(a)}
        if a.size == 1:
            d["value"] = float(a.reshape(-1)[0])
        if a.size <= 32:
            d["values"] = [float(x) for x in a.reshape(-1)]
        prev = self.last_by_id.get(id(t))
        if prev is not None and prev[1].shape == a.shape:
            diff = a - prev[1]
            dh, dm = _enc(_layout(diff))
            d["prev"] = prev[0]
            d["delta"] = dh
            d["dmax"] = dm
            d["changed"] = int((np.abs(diff) > 0).sum())
            d["maxmove"] = float(np.abs(diff).max())
        self.last_by_id[id(t)] = (oid, a)
        self.objects[oid] = d
        self.keys[key] = oid
        return oid

    def value(self, v):
        if isinstance(v, torch.Tensor):
            return {"k": "ref", "ref": self.obj(v)}
        if isinstance(v, bool):
            return {"k": "num", "v": str(v)}
        if isinstance(v, (int, float)):
            return {"k": "num", "v": v}
        if isinstance(v, types.SimpleNamespace):
            return {"k": "model"}
        if isinstance(v, torch.nn.Module):
            return {"k": "layer", "v": f"net.{self.names.get(id(v), type(v).__name__)}"}
        if isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
            return {"k": "tuple", "refs": [self.obj(x) for x in v]}
        return {"k": "str", "v": type(v).__name__}

    def params(self):
        out = {}
        for n, m in self.mods.items():
            for pn in ("weight", "bias"):
                t = getattr(m, pn)
                out[f"{n}.{pn}"] = {"w": self.obj(t), "g": self.obj(t.grad) if t.grad is not None else None}
        return out

    def snap(self, frame, event):
        stack = []
        f = frame
        while f is not None:
            if f.f_code.co_filename == SRC_FILE and f.f_code.co_name in TRACED:
                stack.append(f)
            f = f.f_back
        stack.reverse()
        frames = []
        for fr in stack:
            vs = []
            for k, v in fr.f_locals.items():
                if k.startswith("_") or k in ("p",):
                    continue
                vs.append({"name": k, **self.value(v)})
            frames.append({"func": fr.f_code.co_name, "line": fr.f_lineno, "vars": vs})
        ps = self.params()
        changed_params = {k: v for k, v in ps.items() if self.prev_params.get(k) != v}
        self.prev_params = ps
        self.steps.append({"line": frame.f_lineno, "func": frame.f_code.co_name, "event": event, "frames": frames,
                           "params": changed_params})


def trace(model, image, label, lr):
    ns, mods, names = wrap(model)
    rec = Recorder(model, mods, names)
    for m in mods.values():                    # start clean so the first gradients are visible as new
        m.zero_grad(set_to_none=True)

    def local(frame, event, arg):
        if event in ("line", "return"):
            rec.snap(frame, event)
        return local

    def glob(frame, event, arg):
        if frame.f_code.co_filename == SRC_FILE and frame.f_code.co_name in TRACED:
            return local
        return None

    model.eval()
    sys.settrace(glob)
    try:
        loss = TC.training_step(ns, image, CLASSES.index(label), lr)
    finally:
        sys.settrace(None)
    lines = inspect.getsource(TC).splitlines()
    return {"code": lines, "steps": rec.steps, "objects": rec.objects, "loss": loss, "lr": lr, "label": label,
            "classes": CLASSES}
