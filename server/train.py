"""A training run: choose how many examples of each symbol to take from each source,
train, calibrate, and measure on every test set there is (real digits from MNIST's own
test writers, HASYv2's held-out writers, your own held-out samples, fixed synthetic)."""
import time

import numpy as np

from . import model as M
from . import synth
from .glyphs import CLASSES

SYN_TEST_SEED = 9999


def build(store, spec, seed=0):
    """spec: {"synthetic": n, "mnist": n, "hasy": n, "mine": n | "all", "mine_weight": w} -> X, y, counts."""
    rng = np.random.default_rng(seed)
    Xs, ys, counts = [], [], {}

    def take(X, y, n, name):
        if X is None or not n:
            return
        for k in np.unique(y):
            idx = np.nonzero(y == k)[0]
            if n != "all":
                idx = rng.choice(idx, size=min(int(n), len(idx)), replace=False)
            Xs.append(X[idx])
            ys.append(y[idx])
            counts.setdefault(name, {})[CLASSES[k]] = int(len(idx))

    if spec.get("synthetic"):
        X, y = synth.dataset(int(spec["synthetic"]), seed=seed + 1)
        take(X, y, "all", "synthetic")
    m = store.mnist()
    if m and spec.get("mnist"):
        take(*m["train"], spec["mnist"], "mnist")
    h = store.hasy()
    if h and spec.get("hasy"):
        take(*h["train"], spec["hasy"], "hasy")
    mi = store.mine()
    if mi and spec.get("mine"):
        X, y = mi["train"]
        w = int(spec.get("mine_weight", 5))
        for _ in range(max(1, w)):                             # your own writing counts several times over
            take(X, y, spec["mine"], "mine")
        for c in counts.get("mine", {}):
            counts["mine"][c] //= max(1, w)
    if not Xs:
        raise ValueError("No training examples: choose at least one source with data.")
    return np.concatenate(Xs), np.concatenate(ys), counts


def eval_sets(store):
    out = {}
    m = store.mnist()
    if m:
        X, y = m["test"]
        out["mnist"] = (X[:5000], y[:5000])
    h = store.hasy()
    if h:
        out["hasy"] = h["test"]
    mi = store.mine()
    if mi and len(mi["test"][1]):
        out["mine"] = mi["test"]
    out["synthetic"] = synth.dataset(40, seed=SYN_TEST_SEED)
    return out


def run(store, registry, spec, steps=2000, augment=True, init_from=None, seed=0, name="", progress=lambda f, m, live=None: None,
        should_stop=lambda: False, activate=True):
    t0 = time.time()
    progress(0.01, "Gathering examples")
    X, y, counts = build(store, spec, seed)
    rng = np.random.default_rng(seed + 7)
    val = rng.random(len(y)) < 0.08
    init = None
    if init_from:
        net0, _, _ = registry.load(init_from)
        init = {k: v.clone() for k, v in net0.state_dict().items()}
    net, T, curve = M.train(X[~val], y[~val], X[val], y[val], steps=steps, augment=augment, init=init, seed=seed,
                            progress=lambda f, m, live=None: progress(0.03 + 0.85 * f, m, live), should_stop=should_stop)
    progress(0.92, "Measuring on the test sets")
    ev = {k: M.evaluate(net, T, *v) for k, v in eval_sets(store).items()}
    meta = {"name": name or f"{'+'.join(k for k in counts)} · {len(y)} examples", "spec": spec, "steps": steps, "augment": augment,
            "init_from": init_from, "seed": seed, "counts": counts, "n_train": int((~val).sum()), "curve": curve, "eval": ev,
            "seconds": time.time() - t0}
    meta = registry.save(net, T, meta)
    if activate:
        registry.set_active(meta["id"])
    return meta
