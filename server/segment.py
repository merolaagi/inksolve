"""From ink to symbols in reading order, with the layout that changes meaning.

Input is either pen strokes (the canvas) or a photo. A photo becomes ink components
(connected blobs of dark pixels); after that both are treated the same way, as "pieces"
with a bounding box.

  1. Flat pieces (much wider than tall) are set aside: two stacked flats are "=",
     a flat with a dot above and below is "÷", a long flat with symbols above and
     below is a fraction bar, any other flat is a minus (or the top of a 5 or the
     crossbar of a 7, if it touches one).
  2. Remaining pieces that overlap are one symbol (the two strokes of a +, x, 4 or a
     crossed 7).
  3. Tiny dots low between digits are decimal points.
  4. A symbol that sits clearly higher and smaller than the one before it is an
     exponent; a run of them is one exponent group.
  5. Symbols are grouped into lines (a system of equations is several lines).

The result is a tree the decoder walks: lines of items, where an item is a glyph
(with its own exponent group) or a fraction (numerator items over denominator items).
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from .glyphs import normalize_mask, render_strokes


class Piece:
    def __init__(self, x0, y0, x1, y1, strokes=None, mask=None, origin=None):
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)
        self.strokes = strokes or []          # canvas strokes
        self.mask = mask                      # photo: (mask crop, x0, y0)
        self.origin = origin or []

    w = property(lambda s: max(1.0, s.x1 - s.x0))
    h = property(lambda s: max(1.0, s.y1 - s.y0))
    cx = property(lambda s: (s.x0 + s.x1) / 2)
    cy = property(lambda s: (s.y0 + s.y1) / 2)

    def merge(self, o):
        p = Piece(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1), self.strokes + o.strokes, None,
                  self.origin + o.origin)
        if self.mask is not None or o.mask is not None:
            p.mask = _merge_masks([m for m in (self.mask, o.mask) if m is not None])
        return p


def _merge_masks(ms):
    x0 = min(m[1] for m in ms); y0 = min(m[2] for m in ms)
    x1 = max(m[1] + m[0].shape[1] for m in ms); y1 = max(m[2] + m[0].shape[0] for m in ms)
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    for a, mx, my in ms:
        sl = out[my - y0:my - y0 + a.shape[0], mx - x0:mx - x0 + a.shape[1]]
        np.maximum(sl, a, out=sl)
    return (out, x0, y0)


# ------------------------------------------------------------------ input -> pieces
def pieces_from_strokes(strokes):
    out = []
    for i, s in enumerate(strokes):
        a = np.asarray(s, np.float32).reshape(-1, 2)
        if len(a) == 0:
            continue
        x0, y0 = a.min(0)
        x1, y1 = a.max(0)
        out.append(Piece(x0, y0, x1, y1, [a.tolist()], None, [i]))
    return out


def pieces_from_image(img, max_side=1600):
    """A photo or scan (PIL image) -> ink pieces. Adaptive threshold copes with shading."""
    im = img.convert("L")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    g = np.asarray(im, np.float32) / 255.0
    if g.mean() < 0.45:                                  # light ink on a dark board
        g = 1 - g
    local = ndimage.uniform_filter(g, size=max(15, int(min(g.shape) / 12)))
    ink = g < (local - 0.08)
    ink = ndimage.binary_opening(ink, iterations=1) if ink.sum() > 2000 else ink
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    if n == 0:
        return [], g.shape
    objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(ink, lab, range(1, n + 1))
    big = np.max(sizes) if n else 0
    out = []
    for i, sl in enumerate(objs):
        if sizes[i] < max(12, big * 0.004):              # specks of noise
            continue
        y0, x0 = sl[0].start, sl[1].start
        m = (lab[sl] == i + 1).astype(np.float32)
        # soften with the grey level so anti-aliased photos keep their stroke shape
        m = np.maximum(m * 0.0, np.clip((local[sl] - g[sl]) * 6, 0, 1) * (ndimage.binary_dilation(m > 0, iterations=1)))
        out.append(Piece(x0, y0, sl[1].stop, sl[0].stop, None, (m, x0, y0), [i]))
    return out, g.shape


# ------------------------------------------------------------------ grouping
def _dense(strokes, step=2.0):
    pts = []
    for st in strokes:
        a = np.asarray(st, np.float32).reshape(-1, 2)
        pts.append(a[:1])
        for p, q in zip(a[:-1], a[1:]):
            n = int(np.hypot(*(q - p)) // step)
            if n:
                t = np.linspace(0, 1, n + 2)[1:, None]
                pts.append(p + (q - p) * t)
            else:
                pts.append(q[None])
    return np.concatenate(pts) if pts else np.zeros((0, 2), np.float32)


def ink_gap(a, b):
    """Closest distance between the ink of two stroke pieces (None for photo pieces)."""
    if not a.strokes or not b.strokes:
        return None
    pa, pb = _dense(a.strokes), _dense(b.strokes)
    if len(pa) * len(pb) > 4_000_000:
        pa, pb = pa[::4], pb[::4]
    d = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d.min()))


def _xover(a, b):
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _yover(a, b):
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def segment(pieces):
    """pieces -> {"lines": [[item, ...], ...], "glyphs": [glyph, ...]} (see module doc)."""
    if not pieces:
        return {"lines": [], "glyphs": []}
    hs = [p.h for p in pieces if not (p.w > 2.2 * p.h)]
    top = max(hs) if hs else max(p.w for p in pieces)
    ref_h = float(np.percentile([h for h in hs if h >= 0.5 * top], 75)) if hs else top     # the height of a full-size digit
    # small strokes that touch a bigger one belong to it (the base of a 1, the crossbar of a 7, the top of a 5)
    pieces = sorted(pieces, key=lambda p: max(p.w, p.h))
    pad = 0.06 * ref_h
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(pieces):
            if max(a.w, a.h) >= 0.45 * ref_h:
                continue
            for j in range(len(pieces) - 1, i, -1):
                b = pieces[j]
                if b.w > 2.2 * b.h and b.h < 0.4 * ref_h and max(a.w, a.h) < 0.22 * ref_h:
                    continue                                               # never glue a dot to a bar (÷)
                if max(a.w, a.h) < 0.22 * ref_h and _xover(a, b) < 0.3 * a.w and a.cy > b.y0 + 0.7 * b.h:
                    continue                                               # a decimal point beside a digit stays a point
                g = ink_gap(a, b)
                near = (g < 0.12 * ref_h) if g is not None else (a.x1 + pad >= b.x0 and b.x1 + pad >= a.x0 and a.y1 + pad >= b.y0 and b.y1 + pad >= a.y0)
                if near:
                    pieces[j] = b.merge(a)
                    pieces.pop(i)
                    merged = True
                    break
            if merged:
                break
        pieces.sort(key=lambda p: max(p.w, p.h))
    is_dot = lambda p: max(p.w, p.h) < 0.22 * ref_h
    is_flat = lambda p: (p.w > 2.2 * p.h) and p.h < 0.4 * ref_h and p.w > 0.25 * ref_h
    flats = [p for p in pieces if is_flat(p)]
    dots = [p for p in pieces if is_dot(p) and not is_flat(p)]
    rest = [p for p in pieces if p not in flats and p not in dots]

    glyphs = []                                           # each: {"piece", "kind": "symbol"|"bar"|"point", "fixed": label or None}
    used = set()
    # "=": two flats stacked, similar widths, nothing between them
    for i, a in enumerate(flats):
        if id(a) in used:
            continue
        for b in flats[i + 1:]:
            if id(b) in used:
                continue
            ov = _xover(a, b)
            gap = abs(a.cy - b.cy)
            if ov > 0.55 * min(a.w, b.w) and 0.5 < a.w / b.w < 2 and gap < 0.75 * max(a.w, b.w) and gap < 0.9 * ref_h:
                between = [p for p in rest if min(a.cy, b.cy) < p.cy < max(a.cy, b.cy) and _xover(p, a) > 0.3 * p.w]
                if not between:
                    used.update([id(a), id(b)])
                    glyphs.append({"piece": a.merge(b), "kind": "symbol", "two_flats": True})
                    break
    # "÷": a flat with a dot above and a dot below
    for a in flats:
        if id(a) in used:
            continue
        up = [d for d in dots if id(d) not in used and a.x0 <= d.cx <= a.x1 and d.cy < a.cy and a.cy - d.cy < 0.8 * a.w]
        dn = [d for d in dots if id(d) not in used and a.x0 <= d.cx <= a.x1 and d.cy > a.cy and d.cy - a.cy < 0.8 * a.w]
        if up and dn:
            p = a.merge(up[0]).merge(dn[0])
            used.update([id(a), id(up[0]), id(dn[0])])
            glyphs.append({"piece": p, "kind": "symbol"})
    # fraction bars: long flats with symbols both above and below
    bars = []
    for a in sorted([f for f in flats if id(f) not in used], key=lambda f: -f.w):
        above = [p for p in rest if a.x0 - 0.1 * a.w <= p.cx <= a.x1 + 0.1 * a.w and p.y1 <= a.cy + 0.15 * ref_h and a.cy - p.cy < 1.6 * ref_h]
        below = [p for p in rest if a.x0 - 0.1 * a.w <= p.cx <= a.x1 + 0.1 * a.w and p.y0 >= a.cy - 0.15 * ref_h and p.cy - a.cy < 1.6 * ref_h]
        if above and below and a.w > 0.6 * ref_h:
            used.add(id(a))
            bars.append(a)
    # flats touching a symbol are part of it (top of a 5, crossbar of a 7, the bar of a +)
    for a in flats:
        if id(a) in used:
            continue
        hit = [p for p in rest if _xover(p, a) > 0.25 * min(p.w, a.w) and _yover(p, a) > 0 or
               (_xover(p, a) > 0.5 * a.w and (abs(a.y0 - p.y0) < 0.2 * ref_h or abs(a.y1 - p.y1) < 0.2 * ref_h))]
        if hit:
            i = rest.index(hit[0])
            rest[i] = rest[i].merge(a)
            used.add(id(a))
    # overlapping pieces are one symbol
    changed = True
    while changed:
        changed = False
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                a, b = rest[i], rest[j]
                g = ink_gap(a, b)
                if (g is not None and g < 0.05 * ref_h) or (
                        _xover(a, b) > 0.45 * min(a.w, b.w) and (_yover(a, b) > 0 or min(abs(a.y1 - b.y0), abs(b.y1 - a.y0)) < 0.25 * ref_h)):
                    rest[i] = a.merge(b)
                    rest.pop(j)
                    changed = True
                    break
            if changed:
                break
    for p in rest:
        glyphs.append({"piece": p, "kind": "symbol"})
    for a in flats:
        if id(a) not in used:
            glyphs.append({"piece": a, "kind": "symbol", "flat": True})     # a minus
    for d in dots:
        if id(d) not in used:
            glyphs.append({"piece": d, "kind": "point"})
    for b in bars:
        glyphs.append({"piece": b, "kind": "bar"})
    return _layout(glyphs, ref_h)


def _layout(glyphs, ref_h):
    syms = [g for g in glyphs if g["kind"] == "symbol"]
    bars = [g for g in glyphs if g["kind"] == "bar"]
    points = [g for g in glyphs if g["kind"] == "point"]
    for i, g in enumerate(syms):
        g["i"] = i
    # fractions (widest bar first, so a bar inside a numerator nests)
    owned = set()
    items = []
    for b in sorted(bars, key=lambda g: -g["piece"].w):
        bp = b["piece"]
        num = [g for g in syms if g["i"] not in owned and bp.x0 - 0.1 * bp.w <= g["piece"].cx <= bp.x1 + 0.1 * bp.w and g["piece"].cy < bp.cy and bp.cy - g["piece"].cy < 1.7 * ref_h]
        den = [g for g in syms if g["i"] not in owned and bp.x0 - 0.1 * bp.w <= g["piece"].cx <= bp.x1 + 0.1 * bp.w and g["piece"].cy > bp.cy and g["piece"].cy - bp.cy < 1.7 * ref_h]
        if not num or not den:
            syms.append({**b, "kind": "symbol", "flat": True, "i": len(syms)})
            continue
        owned.update(g["i"] for g in num + den)
        x0 = min([bp.x0] + [g["piece"].x0 for g in num + den]); x1 = max([bp.x1] + [g["piece"].x1 for g in num + den])
        y0 = min(g["piece"].y0 for g in num); y1 = max(g["piece"].y1 for g in den)
        items.append({"type": "frac", "num": _sequence(num, points, ref_h), "den": _sequence(den, points, ref_h),
                      "box": [x0, y0, x1, y1], "bar": [bp.x0, bp.y0, bp.x1, bp.y1]})
    free = [g for g in syms if g["i"] not in owned]
    # lines: full-height items whose vertical extents overlap make a line; smaller items
    # (exponents, operators, lowercase letters) join the nearest line
    entries = [(g["piece"].y0, g["piece"].y1, ("g", g)) for g in free] + [(it["box"][1], it["box"][3], ("f", it)) for it in items]
    tall_e = sorted([e for e in entries if e[1] - e[0] >= 0.75 * ref_h], key=lambda e: e[0]) or sorted(entries, key=lambda e: e[0])
    bands = []
    for y0, y1, e in tall_e:
        sh = 0.12 * (y1 - y0)
        if bands and y0 + sh <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], y1 - sh)
            bands[-1][2].append(e)
        else:
            bands.append([y0 + sh, y1 - sh, [e]])
    for y0, y1, e in entries:
        if any(e is x for b in bands for x in b[2]):
            continue
        c = (y0 + y1) / 2
        best = min(bands, key=lambda b: 0 if b[0] <= c <= b[1] else min(abs(c - b[0]), abs(c - b[1])))
        best[2].append(e)
    lines = [b[2] for b in bands]
    out_lines = []
    for ln in lines:
        gl = [e[1] for e in ln if e[0] == "g"]
        fr = [e[1] for e in ln if e[0] == "f"]
        out_lines.append(_sequence(gl, points, ref_h, fr))
    out_lines = [l for l in out_lines if l]
    order = []

    def walk(seq):
        for it in seq:
            if it["type"] == "glyph":
                order.append(it["glyph"])
                walk(it["sup"])
            elif it["type"] == "frac":
                walk(it["num"])
                walk(it["den"])
    for l in out_lines:
        walk(l)
    for n, g in enumerate(order):                       # number glyphs in reading order
        g["i"] = n
    return {"lines": out_lines, "glyphs": order, "ref_h": ref_h}


def _sequence(gl, points, ref_h, fracs=()):
    """Order items left to right; attach exponents and decimal points."""
    entries = [("g", g["piece"].x0, g) for g in gl] + [("f", f["box"][0], f) for f in fracs]
    entries.sort(key=lambda e: e[1])
    seq = []
    for kind, _, obj in entries:
        if kind == "f":
            seq.append({"type": "frac", **{k: v for k, v in obj.items() if k != "type"}})
            continue
        p = obj["piece"]
        prev = next((s for s in reversed(seq) if s["type"] == "glyph"), None) if seq and seq[-1]["type"] == "glyph" else None
        base = prev
        if base is not None:
            bp = base["glyph"]["piece"]
            raised = (p.cy < bp.y0 + 0.32 * bp.h and p.y1 < bp.y0 + 0.66 * bp.h and p.h < 0.8 * ref_h and p.x0 > bp.x0 + 0.4 * bp.w
                      and not obj.get("two_flats") and not base["glyph"].get("two_flats") and bp.h > 0.3 * ref_h)
            last_sup = base["sup"][-1]["glyph"]["piece"] if base["sup"] else None
            if base["sup"] and last_sup is not None and abs(p.cy - last_sup.cy) < 0.35 * last_sup.h + 0.1 * ref_h and p.y1 < bp.cy and p.h < 0.85 * ref_h:
                base["sup"].append({"type": "glyph", "glyph": obj, "sup": []})
                continue
            if raised and not obj.get("flat"):
                base["sup"].append({"type": "glyph", "glyph": obj, "sup": []})
                continue
        seq.append({"type": "glyph", "glyph": obj, "sup": []})
    # decimal points: a dot low down, between two items of this sequence
    for d in points:
        dp = d["piece"]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a["type"] != "glyph" or b["type"] != "glyph":
                continue
            ap, bp = a["glyph"]["piece"], b["glyph"]["piece"]
            if ap.x1 - 0.2 * ap.w <= dp.cx <= bp.x0 + 0.2 * bp.w and dp.cy > ap.cy:
                seq.insert(i + 1, {"type": "point", "box": [dp.x0, dp.y0, dp.x1, dp.y1]})
                break
    return seq


# ------------------------------------------------------------------ glyph images
def raster(glyph):
    p = glyph["piece"]
    if p.strokes:
        return render_strokes(p.strokes)
    if p.mask is not None:
        return normalize_mask(p.mask[0])
    return np.zeros((28, 28), np.float32)
