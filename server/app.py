"""InkSolve: handwritten maths in, reading and worked answer out.

Run with ./run.sh (port 48631, or the next free one). Everything stays on this computer:
data/ holds downloaded datasets, your own handwriting samples, trained models, the
experiment results and the history of what was read.
"""
import base64
import io
import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import decode as D
from . import inside as IN
from . import experiment as E
from . import mathparse as MP
from . import model as M
from . import segment as S
from . import solve as SV
from . import synth
from . import train as TR
from .data import Store
from .glyphs import CLASSES, to_data_url
from .jobs import Jobs

ROOT = Path(__file__).resolve().parent.parent
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "0.1.0"
DATA = Path(os.environ.get("INKSOLVE_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)

store = Store(DATA)
registry = M.Registry(DATA, ROOT / "models" / "starter.pt")
lab = IN.Lab(registry)
jobs = Jobs(DATA)
HISTORY = DATA / "history.jsonl"
_reads = {}                      # recent readings, so a correction can save the exact mark
_reads_lock = threading.Lock()

app = FastAPI(title="InkSolve", version=VERSION)
PASSWORD = os.environ.get("INKSOLVE_PASSWORD")


@app.middleware("http")
async def guard(request: Request, call_next):
    """Optional password (set INKSOLVE_PASSWORD) for when the app is reachable from elsewhere."""
    if PASSWORD and request.url.path != "/api/health":
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.startswith("Basic "):
            try:
                _, pw = base64.b64decode(auth[6:]).decode().split(":", 1)
                ok = secrets.compare_digest(pw, PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="InkSolve"'})
    return await call_next(request)


def err(msg, code=400):
    raise HTTPException(code, msg)


# ------------------------------------------------------------------ reading
def _decode_image(data_url):
    try:
        b = base64.b64decode(data_url.split(",", 1)[-1])
        return Image.open(io.BytesIO(b))
    except Exception:
        err("That image could not be opened.")


def _model():
    net, T, meta = registry.load()
    if net is None:
        err("There is no trained model yet. Train one on the Train tab.", 409)
    return net, T, meta


def _history_add(rec):
    try:
        with HISTORY.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _history(limit=100):
    if not HISTORY.exists():
        return []
    lines = HISTORY.read_text().splitlines()[-limit:]
    out = []
    for l in reversed(lines):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def _solve_lines(lines, mode):
    trees = []
    for l in lines:
        if not l.get("ok"):
            return None
        trees.append(MP.parse(l["tokens"]))
    try:
        return SV.solve(trees, mode)
    except Exception as e:
        return [{"kind": "error", "note": f"Could not solve this: {e}", "steps": [], "answer": None, "checks": []}]


@app.post("/api/read")
async def read(req: Request):
    body = await req.json()
    mode = body.get("mode", "solve")
    if mode not in ("solve", "grade"):
        err("mode is solve or grade")
    t0 = time.time()
    if body.get("text") is not None:
        text = body["text"].strip()
        if not text:
            err("Type something to solve.")
        lines = []
        for part in [p for p in re.split(r"[\n;]+", text) if p.strip()]:
            try:
                lines.append(D.read_text(part))
            except (MP.ParseError, ValueError, ZeroDivisionError) as e:
                lines.append({"ok": False, "error": f"Couldn't read “{part.strip()}”: {e}", "glyphs": []})
        sol = _solve_lines(lines, mode)
        rec = {"id": uuid.uuid4().hex[:10], "time": time.time(), "source": "typed", "mode": mode, "lines": lines, "solution": sol,
               "glyphs": [], "model": None}
        _history_add({k: rec[k] for k in ("id", "time", "source", "mode")} | {"text": " ; ".join(l.get("text", "?") for l in lines),
                                                                          "answer": _answer_text(sol)})
        return rec
    net, T, meta = _model()
    scale = 1.0
    if body.get("image"):
        im = _decode_image(body["image"])
        pieces, shape = S.pieces_from_image(im)
        scale = im.width / shape[1] if shape[1] else 1.0
        source = "photo"
    else:
        strokes = body.get("strokes") or []
        strokes = [s for s in strokes if s and len(s) >= 1]
        if not strokes:
            err("Draw something first.")
        pieces = S.pieces_from_strokes(strokes)
        source = "drawing"
    if not pieces:
        err("No ink was found. For a photo, try one with dark writing on light paper, filling most of the picture.")
    if len(pieces) > 400:
        err("That picture has too many separate marks to be one problem. Crop it to the maths.")
    layout = S.segment(pieces)
    G = np.stack([S.raster(g) for g in layout["glyphs"]]) if layout["glyphs"] else np.zeros((0, 28, 28), np.float32)
    P = M.predict(net, T, G)
    force = {int(k): v for k, v in (body.get("force") or {}).items() if v in CLASSES}
    lines = D.read(layout, P, mode, force)
    for l in lines:
        l["why"] = D.explain(l, mode)
    glyphs = []
    for g, img, p in zip(layout["glyphs"], G, P):
        pc = g["piece"]
        order = np.argsort(-p)[:3]
        glyphs.append({"i": g["i"], "box": [pc.x0 * scale, pc.y0 * scale, pc.x1 * scale, pc.y1 * scale], "png": to_data_url(img),
                       "top": [{"c": CLASSES[k], "p": float(p[k])} for k in order],
                       "fixed": "=" if g.get("two_flats") else ("-" if g.get("flat") else None), "forced": force.get(g["i"])})
    for l in lines:
        for gi, tok in (l.get("choice") or {}).items():
            glyphs[gi]["read"] = "×" if tok == "*" else tok
        l.pop("choice", None)
    sol = _solve_lines(lines, mode) if all(l.get("ok") for l in lines) and lines else None
    rid = uuid.uuid4().hex[:10]
    with _reads_lock:
        _reads[rid] = {"glyphs": [(g["piece"].strokes, img) for g, img in zip(layout["glyphs"], G)], "source": source}
        while len(_reads) > 50:
            _reads.pop(next(iter(_reads)))
    rec = {"id": rid, "time": time.time(), "source": source, "mode": mode, "lines": lines, "glyphs": glyphs, "solution": sol,
           "model": {"id": meta.get("id"), "name": meta.get("name")}, "ms": int((time.time() - t0) * 1000)}
    if not body.get("preview"):
        _history_add({"id": rid, "time": rec["time"], "source": source, "mode": mode,
                      "text": " ; ".join(l.get("text", "?") for l in lines) or "(nothing read)", "answer": _answer_text(sol),
                      "changed": sum(len(l.get("changed", [])) for l in lines)})
    return rec


def _answer_text(sol):
    if not sol:
        return None
    a = [s.get("answer") or {} for s in sol]
    return " ; ".join(x.get("latex", "") for x in a if x)


@app.post("/api/correct")
async def correct(req: Request):
    """A person says what a mark really is: save it as a sample of their handwriting."""
    b = await req.json()
    rid, gi, label = b.get("read_id"), b.get("glyph"), b.get("label")
    if label not in CLASSES:
        err("Pick one of the symbols InkSolve knows.")
    with _reads_lock:
        r = _reads.get(rid)
    if not r or not isinstance(gi, int) or not 0 <= gi < len(r["glyphs"]):
        err("That reading has expired; read it again first.", 404)
    strokes, img = r["glyphs"][gi]
    rec = store.add_mine(label, strokes=strokes or None, image=None if strokes else img, source="correction")
    return {"saved": rec["id"], "label": label, "status": store.status()["sources"]["mine"]}


# ------------------------------------------------------------------ examples
EXAMPLES = {
    "ambiguous-7": ("6+1=7", {"7": synth.AMBIGUOUS_7}, "6 + 1 = 7, with a 7 that looks like a 1"),
    "linear": ("2x+3=11", {}, "A linear equation"),
    "quadratic": ("x^2-5x+6=0", {}, "A quadratic"),
    "times": ("3x4=12", {}, "x as a times sign"),
    "decimal": ("3.5+1=4.5", {}, "Decimals"),
    "system": ("x+y=10\nx-y=2", {}, "Two equations at once"),
}


@app.get("/api/example/{name}")
def example(name: str, seed: int = None):
    if name not in EXAMPLES:
        err("No such example.", 404)
    text, styles, title = EXAMPLES[name]
    base = seed if seed is not None else int(time.time() * 1000) % 100000

    def make(sd):
        rng = np.random.default_rng(sd)
        out = []
        for row, line in enumerate(text.split("\n")):
            out += synth.expression_strokes(line, rng, height=70, gap=18, x0=40, y0=60 + row * 120, styles=styles, jitter=0.6)
        return out
    strokes = make(base)
    if name == "ambiguous-7" and registry.active_id():
        # pick a version whose last mark the reader in use honestly finds closer to a 1 than a 7
        net, T, _ = registry.load()
        for k in range(60):
            st = make(base + k)
            lay = S.segment(S.pieces_from_strokes(st))
            if len(lay["glyphs"]) != 5:
                continue
            p = M.predict(net, T, [S.raster(lay["glyphs"][-1])])[0]
            if 0.55 <= p[CLASSES.index("1")] <= 0.85:
                strokes = st
                break
    return {"name": name, "title": title, "strokes": [[[round(x, 1), round(y, 1)] for x, y in s] for s in strokes]}


# ------------------------------------------------------------------ your handwriting
@app.post("/api/samples")
async def add_sample(req: Request):
    b = await req.json()
    label = b.get("label")
    strokes = b.get("strokes")
    if b.get("image"):
        im = _decode_image(b["image"]).convert("L")
        a = 255 - np.asarray(im, np.float32)
        from .glyphs import normalize_mask

        rec = store.add_mine(label, image=normalize_mask(a), source="upload")
    else:
        if not strokes:
            err("Draw the symbol first.")
        try:
            rec = store.add_mine(label, strokes=strokes, source="draw")
        except ValueError as e:
            err(str(e))
    return {"saved": rec["id"], "png": _sample_png(rec)}


def _sample_png(r):
    from .glyphs import render_strokes

    a = render_strokes(r["strokes"]) if r.get("strokes") else np.asarray(r["image"], np.float32)
    return to_data_url(a)


@app.get("/api/samples")
def samples(label: str = None, limit: int = 60):
    out = []
    for r in store.mine_samples(label, limit):
        out.append({"id": r["id"], "label": r["label"], "source": r.get("source"), "created": r.get("created"), "png": _sample_png(r)})
    return out


@app.delete("/api/samples/{sid}")
def delete_sample(sid: str):
    if not re.fullmatch(r"[0-9a-f-]+", sid):
        err("Bad id.")
    store.delete_mine(sid)
    return {"ok": True}


@app.get("/api/datasets")
def datasets():
    return store.status()


@app.post("/api/datasets/{name}/download")
def download(name: str):
    if name not in ("mnist", "hasy"):
        err("Unknown dataset.", 404)
    fn = store.download_mnist if name == "mnist" else store.download_hasy
    title = "Download MNIST (real handwritten digits)" if name == "mnist" else "Download HASYv2 (handwritten symbols)"
    return jobs.submit("download", title, lambda progress, stop: fn(lambda f, m: progress(f, m)))


# ------------------------------------------------------------------ training and models
@app.post("/api/train")
async def start_train(req: Request):
    b = await req.json()
    spec = {k: b.get(k) for k in ("synthetic", "mnist", "hasy", "mine", "mine_weight") if b.get(k)}
    if not any(spec.get(k) for k in ("synthetic", "mnist", "hasy", "mine")):
        err("Choose at least one source of examples.")
    steps = int(b.get("steps") or 2000)
    if not 50 <= steps <= 20000:
        err("Steps must be between 50 and 20,000.")
    init = b.get("init_from") or None
    name = (b.get("name") or "").strip()[:80]

    def fn(progress, stop):
        meta = TR.run(store, registry, spec, steps=steps, augment=bool(b.get("augment", True)), init_from=init, seed=int(b.get("seed") or 0),
                      name=name, progress=progress, should_stop=stop)
        return {"id": meta["id"], "name": meta["name"]}
    return jobs.submit("train", f"Train “{name or 'a new model'}”", fn)


@app.get("/api/models")
def models():
    return {"active": registry.active_id(), "models": registry.list()}


@app.get("/api/models/{mid}")
def model_meta(mid: str):
    try:
        return registry.meta(mid)
    except FileNotFoundError:
        err("No such model.", 404)


@app.post("/api/models/{mid}/activate")
def activate(mid: str):
    try:
        registry.set_active(mid)
    except ValueError as e:
        err(str(e), 404)
    return {"active": mid}


@app.delete("/api/models/{mid}")
def delete_model(mid: str):
    try:
        registry.delete(mid)
    except ValueError as e:
        err(str(e))
    return {"ok": True}


# ------------------------------------------------------------------ experiment
@app.post("/api/experiment")
async def start_experiment(req: Request):
    b = await req.json()
    preset = b.get("preset", "quick")
    ns = [int(n) for n in b.get("ns") or [] if 1 <= int(n) <= 5000] or None
    recipes = [r for r in b.get("recipes") or [] if r in E.RECIPES] or None
    steps = int(b["steps"]) if b.get("steps") else None
    return jobs.submit("experiment", f"Experiment: how few examples ({preset})",
                       lambda progress, stop: E.run(store, preset, ns, recipes, steps, progress=progress, should_stop=stop))


@app.get("/api/experiments")
def experiments():
    return E.list_runs(DATA)


# ------------------------------------------------------------------ jobs, history, search, health
@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.get(jid)
    if not j:
        err("No such job.", 404)
    return j


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str):
    j = jobs.stop(jid)
    if not j:
        err("No such job.", 404)
    return j


@app.get("/api/history")
def history(limit: int = 100):
    return _history(limit)


@app.get("/api/search")
def search(q: str = ""):
    words = [w for w in re.split(r"\s+", q.lower().strip()) if w]
    if not words:
        return []

    def score(title, body):
        t, b = title.lower(), (body or "").lower()
        s = 0
        for i, w in enumerate(words):
            last = i == len(words) - 1
            hit_t = w in t if last else re.search(r"(?<!\w)" + re.escape(w), t) is not None
            hit_b = w in b
            if not hit_t and not hit_b:
                return 0
            s += 3 if hit_t else 1
        if " ".join(words) in t:
            s += 4
        return s
    out = []
    seen = set()
    for h in _history(300):
        if h.get("text") in seen:
            continue
        seen.add(h.get("text"))
        sc = score(h.get("text", ""), f"{h.get('answer') or ''} {h.get('source')} {h.get('mode')}")
        if sc:
            out.append((sc, {"group": "History", "title": h.get("text", ""), "sub": f"{h.get('source')} · {h.get('mode')} · "
                             + time.strftime("%d %b %H:%M", time.localtime(h["time"])), "kind": "history", "id": h["id"]}))
    for m in registry.list():
        sc = score(m.get("name", ""), json.dumps(m.get("spec")))
        if sc:
            ev = m.get("eval") or {}
            sub = ", ".join(f"{k} {v['acc'] * 100:.1f}%" for k, v in ev.items() if v)
            out.append((sc, {"group": "Models", "title": m.get("name", ""), "sub": sub, "kind": "model", "id": m["id"]}))
    for x in E.list_runs(DATA)[:20]:
        title = f"Experiment {time.strftime('%d %b %H:%M', time.localtime(x['created']))}"
        sc = score(title + " experiment", " ".join(f"{r['n']} {r['recipe']}" for r in x["rows"]))
        if sc:
            out.append((sc, {"group": "Experiments", "title": title, "sub": f"{len(x['rows'])} runs", "kind": "experiment", "id": x["id"]}))
    out.sort(key=lambda t: -t[0])
    return [o for _, o in out[:20]]


@app.get("/api/health")
def health():
    net, T, meta = registry.load() if registry.active_id() else (None, None, None)
    return {"ok": True, "version": VERSION, "device": M.device(), "model": meta and {"id": meta.get("id"), "name": meta.get("name"),
            "temperature": meta.get("temperature")}, "classes": CLASSES, "jobs": jobs.active(), "password": bool(PASSWORD)}


# ------------------------------------------------------------------ inside the model
_lab_input = {"x": None}


def _lab_x(b):
    import torch

    from .glyphs import render_strokes

    if b.get("strokes"):
        a = render_strokes([s for s in b["strokes"] if s])
    elif b.get("sample"):
        if b["sample"] not in CLASSES:
            err("Unknown symbol.")
        a = IN.sample_input(b["sample"], b.get("seed"))
    elif _lab_input["x"] is not None:
        return _lab_input["x"]
    else:
        a = IN.sample_input("7", 3)
    x = torch.tensor(np.asarray(a, np.float32))[None, None]
    _lab_input["x"] = x
    return x


@app.post("/api/inside/look")
async def inside_look(req: Request):
    """Forward a mark through the scratch copy; with a label also backward; with step also update."""
    b = await req.json()
    label = b.get("label")
    if label is not None and label not in CLASSES:
        err("Unknown symbol.")
    x = _lab_x(b)
    return lab.snapshot(x, label=label, step=bool(b.get("step")) and label is not None)


@app.post("/api/inside/train")
async def inside_train(req: Request):
    b = await req.json()
    steps = max(1, min(200, int(b.get("steps") or 10)))
    batch = max(4, min(256, int(b.get("batch") or 32)))
    return lab.train(steps, batch, store)


@app.post("/api/inside/reset")
async def inside_reset(req: Request):
    b = await req.json()
    start = b.get("start", "trained")
    if start not in ("trained", "random"):
        err("start is trained or random")
    opt = b.get("optimizer")
    if opt not in (None, "sgd", "adam"):
        err("optimizer is sgd or adam")
    return lab.reset(start, opt, b.get("lr"))


@app.post("/api/inside/optimizer")
async def inside_opt(req: Request):
    b = await req.json()
    if b.get("optimizer") not in (None, "sgd", "adam"):
        err("optimizer is sgd or adam")
    lr = b.get("lr")
    if lr is not None and not 1e-5 <= float(lr) <= 2:
        err("Learning rate must be between 0.00001 and 2.")
    lab.set_opt(b.get("optimizer"), lr)
    return lab.state()


@app.post("/api/code/trace")
async def code_trace(req: Request):
    """Run one training step of the Inside copy line by line and return what each line did."""
    from . import tracer

    b = await req.json()
    label = b.get("label") or "7"
    if label not in CLASSES:
        err("Unknown symbol.")
    x = _lab_x(b)
    with lab.lock:
        lab.ensure()
        out = tracer.trace(lab.net, x, label, float(b.get("lr") or lab.cfg["lr"]))
        lab.steps += 1
        lab.history.append({"step": lab.steps, "loss": None, "kind": "traced"})
    out["source"] = lab.source
    return out


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-cache"})
