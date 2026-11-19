"""Reading a line of handwriting: shape evidence times context, as in noisy-channel decoding.

For every mark the symbol model gives calibrated probabilities ("62% a 1, 37% a 7").
A beam search builds whole readings left to right, keeping the best few, and scores each
one as

    log P(shapes | reading)            what the marks look like
  + grammar                            impossible readings ("6 + = 7", "x3") are never kept
  + small dislikes                     a number written 07, two minus signs in a row
  + truth (solve mode only)            a closed statement that is false ("6 + 1 = 1") is
                                       marked down against a true one ("6 + 1 = 7"); an
                                       equation with unknowns ("b + 2 = 8") is neither

The truth term is what turns a mark that looks 62% like a 1 into a 7 when the sum needs a
7. It is exactly what you must NOT do when marking someone's work: a pupil who wrote
6 + 1 = 1 should be read as having written 1. Grading mode therefore drops the truth
term and instead points at marks whose shape alone is unsure, for a person to look at.

The 'x' mark is one class to the symbol model; whether it is the letter x or a times sign
is decided here, by the grammar and the sum ("2x3" can only be 2 × 3; "2x + 3" only 2x).
"""
import math

import numpy as np
import sympy as sp

from . import mathparse as MP
from .glyphs import CLASSES

FALSE_PENALTY = 4.0        # log-odds: in solve mode a false closed statement is taken to be ~55x less likely than a true one
NICE_BONUS = 0.4           # an equation with a tidy answer is a little more likely to be what was meant
BEAM = 96
TOPK = 5
MIN_P = 1e-3


def flatten(seq):
    """Layout items -> stream of ("glyph", g) and ("fixed", token)."""
    out = []
    for it in seq:
        if it["type"] == "glyph":
            out.append(("glyph", it["glyph"]))
            if it["sup"]:
                out.append(("fixed", "^("))
                out += flatten(it["sup"])
                out.append(("fixed", ")^"))
        elif it["type"] == "point":
            out.append(("fixed", "."))
        elif it["type"] == "frac":
            out.append(("fixed", "[/"))
            out += flatten(it["num"])
            out.append(("fixed", "/|/"))
            out += flatten(it["den"])
            out.append(("fixed", "/]"))
    return out


def options(g, probs, force=None):
    """What a glyph could be: [(token, log P(shape), class)]. Layout fixes some marks outright,
    and so does a person who has corrected a mark."""
    f = (force or {}).get(g["i"])
    if f:
        return [("x", 0.0, "x"), ("*", -0.15, "x")] if f == "x" else [(f, 0.0, f)]
    if g.get("two_flats"):
        return [("=", 0.0, "=")]
    if g.get("flat"):
        return [("-", 0.0, "-")]
    p = probs[g["i"]]
    order = np.argsort(-p)
    out = []
    for k in order[:TOPK]:
        if p[k] < MIN_P and out:
            break
        c = CLASSES[k]
        lp = math.log(max(float(p[k]), 1e-9))
        if c == "x":
            out.append(("x", lp, c))
            out.append(("*", lp - 0.15, c))                # the times sign: same mark, decided by context
        elif c == "÷":
            out.append(("÷", lp, c))
        else:
            out.append((c, lp, c))
    return out


class Hyp:
    __slots__ = ("tokens", "shape", "ctx", "state", "choice")

    def __init__(self):
        self.tokens, self.shape, self.ctx, self.state, self.choice = [], 0.0, 0.0, MP.Prefix(), {}

    def ext(self, tok, lp, gi=None):
        st = self.state.copy()
        ok, pen = st.push(tok)
        if not ok:
            return None
        h = Hyp.__new__(Hyp)
        h.tokens = self.tokens + [tok]
        h.shape = self.shape + lp
        h.ctx = self.ctx + pen
        h.state = st
        h.choice = dict(self.choice)
        if gi is not None:
            h.choice[gi] = tok
        return h

    @property
    def score(self):
        return self.shape + self.ctx


def judge(tree):
    """Closed statement -> (True/False, lhs value, rhs value); anything else -> None."""
    if tree[0] != "eq" or MP.variables(tree):
        return None
    try:
        a, b = MP.to_sympy(tree[1]), MP.to_sympy(tree[2])
        return (bool(sp.simplify(a - b) == 0), a, b)
    except Exception:
        return None


def _nice(tree):
    """An equation in one unknown whose answer is a whole number or a simple fraction."""
    vs = MP.variables(tree)
    if tree[0] != "eq" or len(vs) != 1:
        return False
    try:
        e = MP.to_sympy(tree)
        v = MP.SYM[next(iter(vs))]
        sol = sp.solve(sp.Eq(e.lhs, e.rhs), v)
        return bool(sol) and all(s.is_rational and sp.fraction(s)[1] <= 12 for s in sol)
    except Exception:
        return False


def decode_line(stream, probs, mode="solve", n_best=6, force=None):
    beam = [Hyp()]
    for kind, obj in stream:
        nxt = []
        if kind == "fixed":
            for h in beam:
                e = h.ext(obj, 0.0)
                if e:
                    nxt.append(e)
        else:
            opts = options(obj, probs, force)
            for h in beam:
                for tok, lp, _ in opts:
                    e = h.ext(tok, lp, obj["i"])
                    if e:
                        nxt.append(e)
        if not nxt:                                            # nothing fits: keep going on shape alone
            return None
        nxt.sort(key=lambda h: -h.score)
        beam = nxt[:BEAM]
    done = []
    seen = set()
    for h in beam:
        if not h.state.complete():
            continue
        key = tuple(h.tokens)
        if key in seen:
            continue
        seen.add(key)
        try:
            tree = MP.parse(h.tokens)
        except MP.ParseError:
            continue
        truth = judge(tree)
        bonus = 0.0
        if mode == "solve":
            if truth is not None and not truth[0]:
                bonus -= FALSE_PENALTY
            elif truth is None and len(done) < 24 and _nice(tree):
                bonus += NICE_BONUS
        done.append({"h": h, "tree": tree, "truth": truth, "bonus": bonus, "total": h.score + bonus})
    if not done:
        return None
    done.sort(key=lambda d: -d["total"])
    z = np.array([d["total"] for d in done])
    post = np.exp(z - z.max())
    post /= post.sum()
    for d, p in zip(done, post):
        d["posterior"] = float(p)
    return done[:n_best]


def read(layout, probs, mode="solve", force=None):
    """layout from segment.segment(), probs (N,K) from the symbol model -> per-line readings."""
    lines = []
    for li, seq in enumerate(layout["lines"]):
        stream = flatten(seq)
        gids = [o["i"] for k, o in stream if k == "glyph"]
        best = decode_line(stream, probs, mode, force=force)
        if best is None:
            lines.append({"ok": False, "glyphs": gids, "error": "No reading of this line makes sense as maths. "
                          "Correct a mark below, or type it."})
            continue
        top = best[0]
        changed = []
        for gi in gids:
            tok = top["h"].choice.get(gi)
            p = probs[gi]
            shape_best = CLASSES[int(np.argmax(p))]
            sym = {"*": "x", "x": "x"}.get(tok, tok)
            g = layout["glyphs"][gi]
            if tok and sym != shape_best and not g.get("two_flats") and not g.get("flat") and gi not in (force or {}):
                changed.append({"glyph": gi, "shape": shape_best, "shape_p": float(p.max()), "read": tok,
                                "read_p": float(p[CLASSES.index(sym)])})
        lines.append({
            "ok": True, "glyphs": gids, "tokens": top["h"].tokens, "latex": MP.to_latex(top["tree"]),
            "text": MP.to_text(top["tree"]), "confidence": top["posterior"],
            "truth": None if top["truth"] is None else {"true": top["truth"][0], "lhs": str(top["truth"][1]), "rhs": str(top["truth"][2])},
            "changed": changed, "choice": top["h"].choice,
            "alternatives": [{"text": MP.to_text(d["tree"]), "latex": MP.to_latex(d["tree"]), "p": d["posterior"],
                              "truth": None if d["truth"] is None else d["truth"][0]} for d in best[1:5]],
            "unsure": [gi for gi in gids if float(probs[gi].max()) < 0.8 and not layout["glyphs"][gi].get("two_flats")
                       and not layout["glyphs"][gi].get("flat")],
        })
    return lines


def explain(line, mode):
    """One sentence per mark that context changed, in plain words."""
    out = []
    for c in line.get("changed", []):
        rd = "×" if c["read"] == "*" else c["read"]
        why = ("the statement is only true that way" if line.get("truth") and line["truth"]["true"] else
               "it is the only way the line makes sense as maths")
        out.append(f"Mark {c['glyph'] + 1} looks {c['shape_p'] * 100:.0f}% like {c['shape']}, but reads as {rd} "
                   f"({c['read_p'] * 100:.0f}% by shape) because {why}.")
    return out


def read_text(s):
    toks = MP.tokenize_text(s)
    tree = MP.parse(toks)
    t = judge(tree)
    return {"ok": True, "tokens": toks, "latex": MP.to_latex(tree), "text": MP.to_text(tree), "confidence": 1.0,
            "truth": None if t is None else {"true": t[0], "lhs": str(t[1]), "rhs": str(t[2])}, "changed": [], "alternatives": [],
            "unsure": [], "glyphs": []}
