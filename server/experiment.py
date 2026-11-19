"""The training-reduction experiment: how few examples does a reader need, and how much
do augmentation, pretraining on synthetic handwriting, and maths context each save?

For every number of real examples per symbol N and every recipe:
  scratch     train on the N examples as they are
  augment     the same N, each shown bent many ways (slant, rotation, wobble, pen width)
  pretrain    start from a model that has only ever seen synthetic handwriting, then
              fine-tune on the same N, with augmentation

the reader is trained for the same number of steps, then measured on writers it has never
seen:
  symbol accuracy               one mark at a time
  1 -> 7 and 7 -> 1             the confusion this whole project started from
  sums read right, no context   every mark must be right on shape alone (true sums)
  sums read right, with context the decoder may use grammar and whether the sum is true
  grading reads                 all sums, true and false, read exactly as written
  mistakes hidden               false sums ("3 + 4 = 8") that solve mode "corrected" into
                                true ones; grading mode must keep this at zero
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import model as M
from . import synth
from .glyphs import CLASSES, IDX, to_data_url

RECIPES = {"scratch": "From scratch", "augment": "+ augmentation", "pretrain": "+ synthetic pretraining"}
PRESETS = {
    "quick": {"ns": [10, 50, 250], "recipes": ["scratch", "augment", "pretrain"], "steps": 500, "n_expr": 200},
    "full": {"ns": [5, 10, 25, 50, 100, 250, 1000], "recipes": ["scratch", "augment", "pretrain"], "steps": 900, "n_expr": 400},
}


def _pools(store, seed):
    """Training pool and held-out test pool per class: real where we have it, synthetic otherwise."""
    m, h = store.mnist(), store.hasy()
    train, test, src = {}, {}, {}
    syn_tr = synth.dataset(1000, seed=seed + 101)
    syn_te = synth.dataset(60, seed=synth_test_seed())
    for c in CLASSES:
        k = IDX[c]
        if m is not None and c.isdigit():
            train[c] = m["train"][0][m["train"][1] == k]
            test[c] = m["test"][0][m["test"][1] == k][:400]
            src[c] = "MNIST"
        elif h is not None and (h["train"][1] == k).sum() >= 40:
            train[c] = h["train"][0][h["train"][1] == k]
            test[c] = h["test"][0][h["test"][1] == k][:400]
            src[c] = "HASYv2"
        else:
            train[c] = syn_tr[0][syn_tr[1] == k]
            test[c] = syn_te[0][syn_te[1] == k]
            src[c] = "synthetic"
    return train, test, src


def synth_test_seed():
    return 4242


def _pretrained(store, progress, should_stop):
    f = Path(store.dir) / "experiments" / "pretrain-synthetic.pt"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return torch.load(f, map_location="cpu", weights_only=True)
    progress(0.01, "Pretraining once on synthetic handwriting only (kept for later runs)")
    X, y = synth.dataset(300, seed=77)
    net, _, _ = M.train(X, y, steps=1200, augment=True, seed=77, should_stop=should_stop,
                        progress=lambda fr, m, live=None: progress(0.01 + 0.07 * fr, "Pretraining: " + m))
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    torch.save(sd, f)
    return sd


# ------------------------------------------------------------------ expressions made of held-out marks
def make_expressions(test, n, seed):
    """Sums and equations written with real held-out marks. Returns [(kind, [class...], true?)]."""
    rng = np.random.default_rng(seed)
    out = []
    kinds = ["sum", "difference", "product", "equation", "false_sum"]
    for i in range(n):
        k = kinds[i % len(kinds)]
        if k == "sum":
            a, b = rng.integers(0, 10, 2)
            s = f"{a}+{b}={a + b}"
        elif k == "difference":
            a = rng.integers(1, 10); b = rng.integers(0, a + 1)
            s = f"{a}-{b}={a - b}"
        elif k == "product":
            a, b = rng.integers(2, 10, 2)
            s = f"{a}x{b}={a * b}"
        elif k == "equation":
            a = rng.integers(2, 10); b = rng.integers(1, 10); x = rng.integers(1, 10)
            s = f"{a}x+{b}={a * x + b}"
        else:
            a, b = rng.integers(0, 10, 2)
            c = a + b + int(rng.choice([-2, -1, 1, 2, 6]))
            c = c if c >= 0 else a + b + 1
            s = f"{a}+{b}={c}"
        out.append((k, list(s), k != "false_sum"))
    return out


def _render_set(exprs, test, seed):
    rng = np.random.default_rng(seed + 1)
    imgs, owner = [], []
    for e, (_, syms, _) in enumerate(exprs):
        for c in syms:
            pool = test[c]
            imgs.append(pool[rng.integers(len(pool))])
            owner.append(e)
    return np.stack(imgs).astype(np.float32), np.array(owner)


def score_expressions(exprs, P):
    """Read every expression three ways. Returns metrics and a few examples."""
    res = {"n": 0, "true_n": 0, "shape_ok": 0, "context_ok": 0, "grade_ok": 0, "false_n": 0, "hidden_solve": 0, "hidden_grade": 0}
    examples = []
    off = 0
    for kind, syms, true in exprs:
        p = P[off:off + len(syms)]
        gl = [{"i": j} for j in range(len(syms))]
        stream = [("glyph", g) for g in gl]
        off += len(syms)
        shape = [CLASSES[int(k)] for k in p.argmax(1)]
        s_ok = shape == syms
        rs = D.decode_line(stream, p, "solve", n_best=1)
        rg = D.decode_line(stream, p, "grade", n_best=1)
        norm = lambda r: [{"*": "x"}.get(t, t) for t in r[0]["h"].tokens] if r else None
        cs, cg = norm(rs), norm(rg)
        res["n"] += 1
        res["grade_ok"] += cg == syms                          # grading must read what was written, true or not
        if true:
            res["true_n"] += 1
            res["shape_ok"] += s_ok
            res["context_ok"] += cs == syms
        if not true:
            res["false_n"] += 1
            hid = lambda r: bool(r) and r[0]["truth"] is not None and r[0]["truth"][0]
            res["hidden_solve"] += hid(rs)
            res["hidden_grade"] += hid(rg)
        if not s_ok and cs == syms and true and len(examples) < 6:
            examples.append({"truth": "".join(syms), "shape": "".join(shape), "read": "".join(cs), "at": off - len(syms)})
    n, t, f = max(1, res["n"]), max(1, res["true_n"]), max(1, res["false_n"])
    return {"shape_acc": res["shape_ok"] / t, "context_acc": res["context_ok"] / t, "grade_acc": res["grade_ok"] / n,
            "hidden_solve": res["hidden_solve"] / f, "hidden_grade": res["hidden_grade"] / f, "n_expr": res["n"], "n_false": res["false_n"]}, examples


def run(store, preset="quick", ns=None, recipes=None, steps=None, n_expr=None, seed=0, progress=lambda f, m, live=None: None,
        should_stop=lambda: False):
    cfg = dict(PRESETS.get(preset, PRESETS["quick"]))
    cfg.update({k: v for k, v in {"ns": ns, "recipes": recipes, "steps": steps, "n_expr": n_expr}.items() if v})
    t0 = time.time()
    train_pool, test_pool, src = _pools(store, seed)
    pre = _pretrained(store, progress, should_stop) if "pretrain" in cfg["recipes"] else None
    Xt = np.concatenate([test_pool[c] for c in CLASSES])
    yt = np.concatenate([np.full(len(test_pool[c]), IDX[c]) for c in CLASSES])
    exprs = make_expressions(test_pool, cfg["n_expr"], seed + 5)
    Xe, _ = _render_set(exprs, test_pool, seed + 5)
    rows = []
    runs = [(n, r) for n in cfg["ns"] for r in cfg["recipes"]]
    shown = []
    for ri, (n, recipe) in enumerate(runs):
        if should_stop():
            raise InterruptedError("Stopped.")
        rng = np.random.default_rng(seed + n)
        X, y = [], []
        for c in CLASSES:
            pool = train_pool[c]
            idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            X.append(pool[idx])
            y.append(np.full(len(idx), IDX[c]))
        X, y = np.concatenate(X), np.concatenate(y)
        base = 0.08 + 0.9 * ri / len(runs)
        span = 0.9 / len(runs)
        label = f"{n} per symbol, {RECIPES[recipe].lower()}"
        progress(base, f"Run {ri + 1} of {len(runs)}: {label}", {"rows": rows})
        # calibrate on a held-out slice of the same budget when there is one, else on synthetic
        Xc, yc = synth.dataset(8, seed=seed + 900 + n)
        net, T, _ = M.train(X, y, Xc, yc, steps=cfg["steps"], augment=recipe != "scratch", init=pre if recipe == "pretrain" else None,
                            seed=seed + ri, should_stop=should_stop,
                            progress=lambda fr, m, live=None: progress(base + span * fr * 0.85, f"Run {ri + 1} of {len(runs)}: {label}. {m}"))
        ev = M.evaluate(net, T, Xt, yt)
        P = M.predict(net, T, Xe)
        ex, examples = score_expressions(exprs, P)
        p17 = ev["pairs"].get("1|7", {})
        row = {"n": n, "recipe": recipe, "symbol_acc": ev["acc"], "ece": ev["ece"], "one_as_seven": p17.get("a_as_b"),
               "seven_as_one": p17.get("b_as_a"), "digits_acc": float(np.mean([ev["per_class"][c]["acc"] for c in "0123456789" if c in ev["per_class"]])),
               **ex}
        rows.append(row)
        if examples and len(shown) < 6 and recipe == cfg["recipes"][0]:
            for e in examples[:2]:
                e["images"] = [to_data_url(Xe[e["at"] + j]) for j in range(len(e["truth"]))]
                e["n"], e["recipe"] = n, recipe
                shown.append(e)
        progress(base + span, f"Run {ri + 1} of {len(runs)} done: symbols {row['symbol_acc'] * 100:.1f}%, "
                              f"sums {row['shape_acc'] * 100:.0f}% → {row['context_acc'] * 100:.0f}% with context", {"rows": rows})
    out = {"id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4], "created": time.time(), "config": cfg, "sources": src,
           "rows": rows, "examples": shown, "seconds": time.time() - t0, "n_test_symbols": int(len(yt)), "n_expr": len(exprs)}
    d = Path(store.dir) / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{out['id']}.json").write_text(json.dumps(out))
    return out


def list_runs(data_dir):
    d = Path(data_dir) / "experiments"
    out = []
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out
