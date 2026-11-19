"""Inside the model: a scratch copy of the symbol reader that you can poke.

Everything the 3D view shows comes from here, from the real network:
  weights      every number in every layer (about 141,000 of them)
  activations  what each layer produced for the mark you drew (the forward pass)
  gradients    for every weight, which way and how hard the loss wants it to move
  update       how much each weight actually moved in one step
  use          how much each weight contributed to this forward pass: |weight × what came in|

The copy is separate from the reader in use, so training here never changes what the
Solve tab reads with. A step on a single mark freezes BatchNorm statistics and dropout
(eval mode) so it is well defined; batch training runs in the usual training mode.
Gradients are clipped to length 1 so a single mislabelled mark can't wreck the model.
"""
import base64
import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import model as M
from . import synth
from .glyphs import CLASSES

CLIP = 1.0      # gradients longer than this are scaled down, so one odd example can't wreck the model
LAYERS = [  # (name, path into Net, kind)
    ("conv1", ("f", 0, 0), "conv"), ("conv2", ("f", 1, 0), "conv"), ("conv3", ("f", 3, 0), "conv"),
    ("conv4", ("f", 4, 0), "conv"), ("conv5", ("f", 6, 0), "conv"), ("fc1", ("head", 2), "fc"), ("fc2", ("head", 5), "fc"),
]
WHAT = {
    "conv1": "24 filters of 3×3 that look at the raw ink: edges and strokes",
    "conv2": "24 filters, each mixing all 24 maps from conv1",
    "conv3": "48 filters on the half-size maps: corners, curves, crossings",
    "conv4": "48 filters mixing those",
    "conv5": "64 filters on the quarter-size maps: parts of symbols",
    "fc1": "128 neurons, each weighing all 576 numbers left after the convolutions",
    "fc2": "22 outputs, one per symbol, each weighing the 128 neurons",
}


def _mod(net, path):
    m = net
    for p in path:
        m = getattr(m, p) if isinstance(p, str) else m[p]
    return m


def _f16(a):
    return base64.b64encode(np.asarray(a, np.float16).tobytes()).decode()


def _u8(a):
    a = np.asarray(a, np.float32)
    hi = float(a.max()) if a.size else 0.0
    q = (np.clip(a / hi, 0, 1) * 255).astype(np.uint8) if hi > 0 else np.zeros(a.shape, np.uint8)
    return base64.b64encode(q.tobytes()).decode(), hi


class Lab:
    def __init__(self, registry):
        self.registry = registry
        self.lock = threading.RLock()
        self.net = None
        self.opt = None
        self.cfg = {"optimizer": "sgd", "lr": 0.02, "start": "trained"}
        self.history = []
        self.steps = 0
        self.last = None          # last forward input (1,1,28,28)
        self.last_label = None
        self.rng = np.random.default_rng(0)

    # -------------------------------------------------------------- setup
    def reset(self, start="trained", optimizer=None, lr=None):
        with self.lock:
            self.cfg["start"] = start
            if optimizer:
                self.cfg["optimizer"] = optimizer
            if lr:
                self.cfg["lr"] = float(lr)
            elif self.cfg["optimizer"] == "sgd":
                self.cfg["lr"] = 0.02 if start == "trained" else 0.1     # an untrained net can take bigger steps
            net = M.Net()
            if start == "trained":
                src, _, meta = self.registry.load()
                if src is not None:
                    net.load_state_dict(copy.deepcopy(src.state_dict()))
                self.source = (meta or {}).get("name", "trained reader")
            else:
                torch.manual_seed(int(time.time()) % 10000)
                net = M.Net()
                # untrained: BatchNorm has never seen data, so use identity statistics
                self.source = "random weights (untrained)"
            net.eval()
            self.net = net
            self._make_opt()
            self.history, self.steps = [], 0
        return self.state()

    def _make_opt(self):
        params = [p for p in self.net.parameters()]
        if self.cfg["optimizer"] == "adam":
            self.opt = torch.optim.Adam(params, lr=self.cfg["lr"] / 20)
        else:
            self.opt = torch.optim.SGD(params, lr=self.cfg["lr"])

    def set_opt(self, optimizer=None, lr=None):
        with self.lock:
            if optimizer:
                self.cfg["optimizer"] = optimizer
            if lr:
                self.cfg["lr"] = float(lr)
            self._make_opt()

    def ensure(self):
        if self.net is None:
            self.reset("trained")

    # -------------------------------------------------------------- reading the network
    def weights(self):
        return {n: _mod(self.net, p).weight.detach().cpu().numpy().copy() for n, p, _ in LAYERS}

    def counts(self):
        total = sum(p.numel() for p in self.net.parameters())
        shown = sum(_mod(self.net, p).weight.numel() for _, p, _ in LAYERS)
        return {"total": int(total), "shown": int(shown), "other": int(total - shown)}

    def forward(self, x):
        """x (1,1,28,28) -> activations per layer (after ReLU/pool), logits; keeps the graph."""
        acts = {}
        h = x
        f = self.net.f
        h = f[0](h); acts["conv1"] = h
        h = f[1](h); acts["conv2"] = h
        h = f[2](h); acts["pool1"] = h
        h = f[3](h); acts["conv3"] = h
        h = f[4](h); acts["conv4"] = h
        h = f[5](h); acts["pool2"] = h
        h = f[6](h); acts["conv5"] = h
        h = f[7](h); acts["pool3"] = h
        hd = self.net.head
        flat = hd[0](h)
        z = hd[2](hd[1](flat)); a = hd[3](z); acts["fc1"] = a
        logits = hd[5](hd[4](a)); acts["fc2"] = logits
        inputs = {"conv1": x, "conv2": acts["conv1"], "conv3": acts["pool1"], "conv4": acts["conv3"], "conv5": acts["pool2"],
                  "fc1": flat, "fc2": a}
        return acts, inputs, logits

    @staticmethod
    def use(name, kind, w, inp):
        """How much each weight contributed to this pass: |w| × mean |input it multiplies|."""
        a = inp.detach().abs().cpu().numpy()[0]
        if kind == "fc":
            return np.abs(w) * a[None, :]
        # conv: weight[o, i, ky, kx] multiplies channel i; use the mean of that channel's input
        m = a.reshape(a.shape[0], -1).mean(1)
        return np.abs(w) * m[None, :, None, None]

    def snapshot(self, x, label=None, step=False, batch=None):
        """Forward the mark; if label is given, backward too; if step, apply one update."""
        with self.lock:
            self.ensure()
            for p in self.net.parameters():
                p.grad = None
            acts, inputs, logits = self.forward(x)
            probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
            out = {"probs": probs.round(4).tolist(), "classes": CLASSES, "source": self.source, "cfg": dict(self.cfg),
                   "counts": self.counts(), "steps": self.steps}
            w_before = self.weights()
            loss = None
            if label is not None:
                y = torch.tensor([CLASSES.index(label)])
                if batch is not None:
                    xb, yb = batch
                    lg = self.net(torch.cat([x, xb]))
                    loss = F.cross_entropy(lg, torch.cat([y, yb]))
                else:
                    loss = F.cross_entropy(logits, y)
                loss.backward()
                out["loss"] = float(loss.item())
            layers = []
            grads = {}
            for name, path, kind in LAYERS:
                m = _mod(self.net, path)
                w = w_before[name]
                g = m.weight.grad.detach().cpu().numpy() if m.weight.grad is not None else None
                grads[name] = g
                u = self.use(name, kind, w, inputs[name])
                layers.append({"name": name, "kind": kind, "shape": list(w.shape), "what": WHAT[name], "w": _f16(w),
                               "g": _f16(g) if g is not None else None, "use": _u8(u)[0],
                               "stats": {"n": int(w.size), "mean_abs_w": float(np.abs(w).mean()), "max_abs_w": float(np.abs(w).max()),
                                         "mean_abs_g": float(np.abs(g).mean()) if g is not None else None,
                                         "use_share": float(u.sum())}})
            tot = sum(l["stats"]["use_share"] for l in layers) or 1
            for l in layers:
                l["stats"]["use_share"] /= tot
            if step and label is not None:
                out["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP))
                self.opt.step()
                self.steps += 1
                w_after = self.weights()
                with torch.no_grad():
                    after = F.softmax(self.net(x), 1)[0].cpu().numpy()
                out["probs_after"] = after.round(4).tolist()
                for l in layers:
                    d = w_after[l["name"]] - w_before[l["name"]]
                    l["d"] = _f16(d)
                    l["stats"]["mean_abs_d"] = float(np.abs(d).mean())
                    l["stats"]["max_abs_d"] = float(np.abs(d).max())
                    l["stats"]["changed"] = float((np.abs(d) > 1e-7).mean())
                    l["stats"]["rel_change"] = float(np.abs(d).sum() / (np.abs(w_before[l["name"]]).sum() or 1))
                self.history.append({"step": self.steps, "loss": out["loss"], "kind": "one mark"})
            out["layers"] = layers
            out["acts"] = self._acts(acts)
            out["input"] = _u8(x.detach().cpu().numpy()[0, 0])[0]
            out["history"] = self.history[-300:]
            return out

    @staticmethod
    def _acts(acts):
        out = []
        for k in ("conv1", "conv2", "pool1", "conv3", "conv4", "pool2", "conv5", "pool3", "fc1", "fc2"):
            a = acts[k].detach().cpu().numpy()[0]
            if k == "fc2":
                a = F.softmax(torch.tensor(a), 0).numpy()
            data, hi = _u8(np.maximum(a, 0))
            out.append({"name": k, "shape": list(a.shape), "data": data, "max": hi})
        return out

    # -------------------------------------------------------------- training loop
    def train(self, steps=10, batch=32, store=None):
        """Ordinary minibatch training on synthetic handwriting (plus yours and MNIST if present)."""
        with self.lock:
            self.ensure()
            X, y = synth.dataset(max(2, batch // len(CLASSES) + 1), seed=int(self.rng.integers(1 << 30)))
            pools = [(X, y)]
            if store is not None:
                mi = store.mine()
                if mi:
                    pools.append(mi["train"])
                m = store.mnist()
                if m:
                    idx = self.rng.choice(len(m["train"][1]), size=batch, replace=False)
                    pools.append((m["train"][0][idx], m["train"][1][idx]))
            Xa = np.concatenate([p[0] for p in pools]).astype(np.float32)
            ya = np.concatenate([p[1] for p in pools]).astype(np.int64)
            gen = torch.Generator().manual_seed(int(self.rng.integers(1 << 30)))
            losses, accs = [], []
            w0 = self.weights()
            self.net.train()                      # batches: BatchNorm learns its statistics as usual
            for _ in range(int(steps)):
                idx = self.rng.choice(len(ya), size=min(batch, len(ya)), replace=False)
                xb = M.augment_batch(torch.tensor(Xa[idx]).unsqueeze(1), gen, 0.6)
                yb = torch.tensor(ya[idx])
                lg = self.net(xb)
                loss = F.cross_entropy(lg, yb)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
                self.opt.step()
                self.steps += 1
                losses.append(float(loss.item()))
                accs.append(float((lg.argmax(1) == yb).float().mean()))
                self.history.append({"step": self.steps, "loss": losses[-1], "acc": accs[-1], "kind": "batch"})
            self.net.eval()
            w1 = self.weights()
            moved = {n: float(np.abs(w1[n] - w0[n]).mean()) for n in w0}
            return {"steps": self.steps, "loss": float(np.mean(losses)), "acc": float(np.mean(accs)), "moved": moved,
                    "history": self.history[-300:], "seen": int(steps) * min(batch, len(ya))}

    def state(self):
        return {"source": getattr(self, "source", None), "cfg": dict(self.cfg), "steps": self.steps,
                "counts": self.counts() if self.net is not None else None}


def sample_input(label="7", seed=None):
    rng = np.random.default_rng(seed)
    return synth.sample(label, rng)
