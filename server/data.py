"""Where training examples come from, all normalised to 28x28 by glyphs.py.

  synthetic  generated on demand (synth.py): every class, any number, offline.
  mnist      real handwritten digits, 60,000 for training and 10,000 for testing, from
             about 500 writers. Downloaded once (about 11 MB) from a mirror list.
  hasy       HASYv2: 168,000 handwritten symbols (LaTeX classes) from many writers,
             downloaded once (about 32 MB) from Zenodo; the classes InkSolve uses are kept
             (operators, parentheses, letters, digits). Split by writer, so the test
             writers are never seen in training.
  mine       your own handwriting: symbols you draw on the Data tab, and marks you
             correct on the Solve tab. Kept as strokes (or a cut-out for photos), so
             improvements to rendering apply to old samples too.
"""
import csv
import gzip
import io
import json
import struct
import tarfile
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .glyphs import CLASSES, IDX, normalize_mask, render_strokes

MNIST_MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/",
                 "https://raw.githubusercontent.com/fgnt/mnist/master/"]
MNIST_FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
               "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
HASY_URLS = ["https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1", "https://zenodo.org/records/259444/files/HASYv2.tar.bz2"]
# HASY's LaTeX labels -> InkSolve classes (whatever is present is used)
HASY_MAP = {"+": "+", "-": "-", "=": "=", "\\times": "x", "x": "x", "y": "y", "a": "a", "b": "b", "z": "z",
            "(": "(", ")": ")", "\\div": "÷", "/": "/", "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
            "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "\\mathds{1}": None}


class Store:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        (self.dir / "datasets").mkdir(parents=True, exist_ok=True)
        (self.dir / "mine").mkdir(parents=True, exist_ok=True)
        self._cache = {}

    # -------------------------------------------------------------- downloads
    def _get(self, urls, progress, label):
        last = None
        for u in urls:
            try:
                progress(0.02, f"Downloading {label} from {u.split('/')[2]}")
                req = urllib.request.Request(u, headers={"User-Agent": "InkSolve/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    total = int(r.headers.get("Content-Length") or 0)
                    buf, got = io.BytesIO(), 0
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        buf.write(chunk)
                        got += len(chunk)
                        if total:
                            progress(0.05 + 0.85 * got / total, f"Downloading {label}: {got // 1_000_000} of {total // 1_000_000} MB")
                    return buf.getvalue()
            except Exception as e:  # try the next mirror
                last = e
        raise RuntimeError(f"Could not download {label}: {last}")

    def download_mnist(self, progress=lambda f, m: None):
        d = self.dir / "datasets" / "mnist"
        d.mkdir(parents=True, exist_ok=True)
        for i, (k, f) in enumerate(MNIST_FILES.items()):
            if (d / f).exists():
                continue
            blob = self._get([m + f for m in MNIST_MIRRORS], lambda fr, m: progress((i + fr) / 4, m), f"MNIST {k}")
            (d / f).write_bytes(blob)
        self._cache.pop("mnist", None)
        return self.status()

    def download_hasy(self, progress=lambda f, m: None):
        d = self.dir / "datasets" / "hasy"
        d.mkdir(parents=True, exist_ok=True)
        blob = self._get(HASY_URLS, progress, "HASYv2")
        progress(0.92, "Unpacking the symbols InkSolve uses")
        keep = []
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:bz2") as tf:
            names = {m.name: m for m in tf.getmembers()}
            lab = next(n for n in names if n.endswith("hasy-data-labels.csv"))
            rows = list(csv.DictReader(io.TextIOWrapper(tf.extractfile(names[lab]), "utf-8")))
            root = lab.rsplit("/", 1)[0] + "/" if "/" in lab else ""
            X, y, users = [], [], []
            for r in rows:
                c = HASY_MAP.get(r["latex"])
                if not c:
                    continue
                m = names.get(root + r["path"]) or names.get(r["path"])
                if m is None:
                    continue
                im = Image.open(tf.extractfile(m)).convert("L")
                X.append(normalize_mask(255 - np.asarray(im)))
                y.append(IDX[c])
                users.append(int(r.get("user_id") or 0))
                keep.append(c)
        if not X:
            raise RuntimeError("HASYv2 downloaded, but none of its labels matched InkSolve's symbols.")
        np.savez_compressed(d / "hasy.npz", X=np.stack(X).astype(np.float32), y=np.array(y), users=np.array(users))
        self._cache.pop("hasy", None)
        return self.status()

    # -------------------------------------------------------------- loading
    def mnist(self):
        if "mnist" in self._cache:
            return self._cache["mnist"]
        d = self.dir / "datasets" / "mnist"
        if not all((d / f).exists() for f in MNIST_FILES.values()):
            return None

        def idx(path):
            with gzip.open(path) as f:
                magic = struct.unpack(">I", f.read(4))[0]
                dims = [struct.unpack(">I", f.read(4))[0] for _ in range(magic & 0xFF)]
                return np.frombuffer(f.read(), np.uint8).reshape(dims)
        out = {"train": (idx(d / MNIST_FILES["train_x"]).astype(np.float32) / 255, idx(d / MNIST_FILES["train_y"]).astype(np.int64)),
               "test": (idx(d / MNIST_FILES["test_x"]).astype(np.float32) / 255, idx(d / MNIST_FILES["test_y"]).astype(np.int64))}
        self._cache["mnist"] = out
        return out

    def hasy(self):
        if "hasy" in self._cache:
            return self._cache["hasy"]
        f = self.dir / "datasets" / "hasy" / "hasy.npz"
        if not f.exists():
            return None
        z = np.load(f)
        X, y, u = z["X"], z["y"], z["users"]
        test_users = set(np.unique(u)[::5])                      # one writer in five is only ever seen in testing
        te = np.isin(u, list(test_users))
        out = {"train": (X[~te], y[~te]), "test": (X[te], y[te])}
        self._cache["hasy"] = out
        return out

    # -------------------------------------------------------------- your own handwriting
    def add_mine(self, label, strokes=None, image=None, source="draw"):
        if label not in IDX:
            raise ValueError(f"Unknown symbol {label!r}.")
        if not strokes and image is None:
            raise ValueError("Give strokes or an image.")
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        rec = {"id": sid, "label": label, "source": source, "created": time.time()}
        if strokes:
            rec["strokes"] = strokes
        else:
            rec["image"] = np.asarray(image, np.float32).round(3).tolist()
        d = self.dir / "mine" / _safe(label)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sid}.json").write_text(json.dumps(rec))
        self._cache.pop("mine", None)
        return rec

    def mine(self):
        if "mine" in self._cache:
            return self._cache["mine"]
        X, y, ids = [], [], []
        for f in sorted((self.dir / "mine").glob("*/*.json")):
            try:
                r = json.loads(f.read_text())
                a = render_strokes(r["strokes"]) if r.get("strokes") else np.asarray(r["image"], np.float32)
            except Exception:
                continue
            X.append(a)
            y.append(IDX[r["label"]])
            ids.append(r["id"])
        if not X:
            out = None
        else:
            X, y = np.stack(X).astype(np.float32), np.array(y, np.int64)
            rng = np.random.default_rng(0)
            te = rng.random(len(y)) < 0.2                      # one sample in five is kept for testing
            out = {"train": (X[~te], y[~te]), "test": (X[te], y[te]), "all": (X, y), "ids": ids}
        self._cache["mine"] = out
        return out

    def mine_samples(self, label=None, limit=60):
        out = []
        for f in sorted((self.dir / "mine").glob(f"{_safe(label) if label else '*'}/*.json"), reverse=True)[:limit]:
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            out.append(r)
        return out

    def delete_mine(self, sid):
        for f in (self.dir / "mine").glob(f"*/{sid}.json"):
            f.unlink()
        self._cache.pop("mine", None)

    # -------------------------------------------------------------- what is here
    def status(self):
        counts = {c: {} for c in CLASSES}
        srcs = {}
        m = self.mnist()
        srcs["mnist"] = {"present": m is not None, "train": int(len(m["train"][1])) if m else 0, "test": int(len(m["test"][1])) if m else 0}
        if m:
            for k, v in zip(*np.unique(m["train"][1], return_counts=True)):
                counts[CLASSES[k]]["mnist"] = int(v)
        h = self.hasy()
        srcs["hasy"] = {"present": h is not None, "train": int(len(h["train"][1])) if h else 0, "test": int(len(h["test"][1])) if h else 0}
        if h:
            for k, v in zip(*np.unique(h["train"][1], return_counts=True)):
                counts[CLASSES[k]]["hasy"] = int(v)
        mi = self.mine()
        srcs["mine"] = {"present": mi is not None, "train": int(len(mi["train"][1])) if mi else 0, "test": int(len(mi["test"][1])) if mi else 0}
        if mi:
            for k, v in zip(*np.unique(mi["all"][1], return_counts=True)):
                counts[CLASSES[k]]["mine"] = int(v)
        srcs["synthetic"] = {"present": True, "train": None, "test": None}
        return {"sources": srcs, "counts": counts}


def _safe(label):
    return {"/": "slash", "÷": "div", "(": "lparen", ")": "rparen", "+": "plus", "-": "minus", "=": "eq"}.get(label, label)
