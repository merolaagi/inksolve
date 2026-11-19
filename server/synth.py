"""Synthetic handwriting: every symbol as a few pen strokes, in several styles people
actually use (a 7 with or without a crossbar, a 1 with a flag or a base, open and closed
4s, a 2 with a loop), then bent the way a hand bends them: slant, rotation, size,
wobble, a shaky line, and a pen of varying width.

It is no substitute for real handwriting (MNIST, HASYv2, your own), but it covers every
symbol from the first run, works offline, and lets the experiment control how
ambiguous a 1 and a 7 are allowed to look.
"""
import math

import numpy as np

from .glyphs import CLASSES, render_strokes


def _ellipse(cx, cy, rx, ry, a0=-90, a1=270, n=11):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]


# Each class: a list of styles; each style a list of strokes; each stroke a list of points in a unit box (y down).
T = {
    "0": [[_ellipse(.5, .5, .3, .45, -100, 265)], [_ellipse(.5, .5, .26, .46, -80, 290)]],
    "1": [[[(.5, .05), (.5, .95)]],
          [[(.33, .24), (.52, .05), (.52, .95)]],
          [[(.33, .24), (.52, .05), (.52, .95)], [(.3, .95), (.74, .95)]]],
    "2": [[[(.2, .28), (.3, .1), (.55, .04), (.76, .15), (.76, .35), (.55, .6), (.2, .95), (.82, .95)]],
          [[(.22, .25), (.4, .06), (.68, .1), (.74, .32), (.4, .75), (.25, .93), (.35, .85), (.5, .88), (.82, .95)]]],
    "3": [[[(.22, .15), (.45, .04), (.72, .14), (.7, .35), (.45, .48), (.72, .6), (.75, .82), (.5, .96), (.22, .86)]],
          [[(.22, .08), (.75, .08), (.45, .42), (.72, .55), (.75, .8), (.5, .96), (.22, .86)]]],
    "4": [[[(.62, .05), (.14, .66), (.86, .66)], [(.64, .32), (.64, .96)]],
          [[(.3, .05), (.22, .58), (.8, .58)], [(.66, .05), (.66, .96)]]],
    "5": [[[(.76, .05), (.32, .05), (.28, .45), (.55, .38), (.76, .55), (.76, .8), (.5, .96), (.24, .88)]],
          [[(.32, .05), (.28, .45), (.55, .38), (.76, .55), (.76, .8), (.5, .96), (.24, .88)], [(.32, .05), (.78, .05)]]],
    "6": [[[(.7, .07), (.45, .15), (.27, .45), (.25, .76), (.4, .96), (.66, .93), (.76, .72), (.6, .55), (.4, .55), (.27, .68)]],
          [[(.62, .04), (.3, .5), (.26, .78), (.45, .96), (.7, .86), (.68, .6), (.45, .55), (.3, .7)]]],
    "7": [[[(.2, .07), (.82, .07), (.45, .96)]],
          [[(.2, .07), (.82, .07), (.45, .96)], [(.35, .52), (.74, .52)]],
          [[(.2, .1), (.8, .06), (.56, .5), (.46, .96)]],
          [[(.36, .14), (.66, .05), (.56, .5), (.5, .96)]]],                       # short top: looks like a flagged 1
    "8": [[[(.62, .08), (.36, .1), (.3, .28), (.5, .46), (.72, .63), (.7, .86), (.5, .96), (.3, .86), (.3, .65), (.5, .48), (.7, .3), (.66, .1), (.62, .08)]],
          [_ellipse(.5, .27, .2, .21, 90, 450), _ellipse(.5, .72, .24, .24, -90, 270)]],
    "9": [[[(.72, .3), (.55, .07), (.3, .12), (.27, .36), (.46, .49), (.72, .34), (.72, .1), (.7, .6), (.6, .96)]],
          [[(.72, .26), (.52, .06), (.3, .16), (.3, .4), (.52, .48), (.72, .32)], [(.72, .1), (.72, .96)]]],
    "+": [[[(.12, .5), (.88, .5)], [(.5, .12), (.5, .88)]], [[(.5, .14), (.5, .86)], [(.14, .52), (.86, .48)]]],
    "-": [[[(.1, .5), (.9, .5)]]],
    "=": [[[(.12, .36), (.88, .36)], [(.12, .64), (.88, .64)]]],
    "x": [[[(.2, .2), (.8, .8)], [(.8, .2), (.2, .8)]],
          [[(.22, .25), (.38, .2), (.62, .8), (.8, .76)], [(.78, .22), (.22, .8)]]],
    "y": [[[(.2, .08), (.6, .6)], [(.82, .08), (.36, .96)]],
          [[(.2, .1), (.24, .5), (.48, .56), (.76, .44), (.78, .1), (.76, .8), (.56, .96), (.32, .9)]]],
    "a": [[[(.76, .36), (.55, .25), (.3, .36), (.25, .66), (.45, .82), (.7, .66), (.76, .36), (.76, .82), (.84, .88)]],
          [_ellipse(.47, .55, .26, .27, -30, 330), [(.73, .3), (.73, .85)]]],
    "b": [[[(.3, .02), (.3, .96)], [(.3, .62), (.5, .46), (.72, .56), (.74, .8), (.52, .96), (.3, .9)]],
          [[(.32, .02), (.3, .94), (.5, .96), (.72, .82), (.7, .58), (.5, .5), (.31, .62)]]],
    "z": [[[(.18, .15), (.82, .15), (.18, .86), (.84, .86)]],
          [[(.18, .15), (.82, .15), (.18, .86), (.84, .86)], [(.32, .5), (.7, .5)]]],
    "(": [[[(.66, .03), (.4, .28), (.34, .5), (.4, .72), (.66, .97)]]],
    ")": [[[(.34, .03), (.6, .28), (.66, .5), (.6, .72), (.34, .97)]]],
    "÷": [[[(.12, .5), (.88, .5)], [(.5, .2), (.51, .21)], [(.5, .8), (.51, .81)]]],
    "/": [[[(.78, .04), (.22, .96)]]],
}
# width/height of the unit box each class is drawn in (a minus is wide and flat, a 1 tall and thin)
ASPECT = {"-": (1.0, .12), "=": (1.0, .55), "÷": (1.0, .9), "+": (1.0, 1.0), "x": (.8, .8), "a": (.8, .75), "z": (.8, .8),
          "1": (.45, 1.0), "(": (.4, 1.0), ")": (.4, 1.0), "/": (.6, 1.0), "y": (.75, 1.0)}


def _smooth(stroke, n=40):
    """Catmull-Rom through the control points."""
    p = np.asarray(stroke, np.float64)
    if len(p) < 3:
        t = np.linspace(0, 1, max(2, n // 3))[:, None]
        return p[0] + (p[-1] - p[0]) * t if len(p) == 2 else p
    P = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, max(3, n // (len(p) - 1)), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-1])
    return np.array(out)


# A 7 with only a tiny flag on top: a real habit, and the classic 1-or-7 puzzle. Used for the
# demo on the Solve tab, never for training (so the reader stays honestly unsure about it).
AMBIGUOUS_7 = [[(.40, .14), (.60, .05), (.55, .5), (.52, .96)]]


def strokes_for(label, rng, style=None, jitter=1.0):
    """Random pen strokes for one symbol, in unit-ish coordinates. style may be an index or a template."""
    styles = T[label]
    st = style if isinstance(style, list) else styles[style if style is not None else rng.integers(len(styles))]
    aw, ah = ASPECT.get(label, (.7, 1.0))
    slant = rng.normal(0, .18) * jitter
    rot = rng.normal(0, .07) * jitter
    sx, sy = aw * (1 + rng.normal(0, .08) * jitter), ah * (1 + rng.normal(0, .08) * jitter)
    c, s = math.cos(rot), math.sin(rot)
    out = []
    for stroke in st:
        q = np.array([(x + rng.normal(0, .035) * jitter, y + rng.normal(0, .035) * jitter) for x, y in stroke])
        q = _smooth(q)
        q[:, 0] = (q[:, 0] - .5) * sx
        q[:, 1] = (q[:, 1] - .5) * sy
        q[:, 0] += -slant * q[:, 1]
        q = q @ np.array([[c, s], [-s, c]])
        q += rng.normal(0, .006 * jitter, q.shape)                 # a shaky hand
        out.append(q)
    return [s.tolist() for s in out]


def sample(label, rng, jitter=1.0):
    width = float(rng.uniform(1.5, 3.1))
    return render_strokes(strokes_for(label, rng, jitter=jitter), width=width)


def dataset(n_per_class, seed=0, classes=CLASSES, jitter=1.0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, c in enumerate(classes):
        for _ in range(n_per_class):
            X.append(sample(c, rng, jitter))
            y.append(CLASSES.index(c))
    return np.stack(X).astype(np.float32), np.array(y, np.int64)


def expression_strokes(text, rng, height=60, gap=14, x0=20, y0=40, styles=None, jitter=1.0):
    """Lay out a typed expression as handwriting on a canvas: one synthetic glyph per character.
    '^' raises the run of letters and digits that follows. Used for examples and for testing the reader end to end."""
    strokes, x, raise_next = [], x0, False
    for ch in text:
        if ch == " ":
            x += gap
            continue
        if ch == "^":
            raise_next = True
            continue
        if ch == ".":
            d = float(rng.uniform(2.5, 4))
            strokes.append([[x - gap * .5, y0 + height - d], [x - gap * .5 + d * .6, y0 + height - d * .4]])
            x += gap * .4
            continue
        lab = {"*": "x", "×": "x"}.get(ch, ch)
        if raise_next and not ch.isalnum():
            raise_next = False
        st = strokes_for(lab, rng, style=(styles or {}).get(lab), jitter=jitter)
        aw, ah = ASPECT.get(lab, (.7, 1.0))
        h = height * (.55 if raise_next else 1.0)
        lowercase = lab in ("a", "x", "z") and not raise_next
        top = y0 - (height * .45 if raise_next else 0) + (height * .2 if lowercase else 0)
        hh = h * (.8 if lowercase else 1.0)
        for s in st:
            s = np.asarray(s)
            strokes.append([[float(x + (px + .5) * hh), float(top + (py + .5) * hh)] for px, py in s])
        x += aw * hh + (gap * .6 if raise_next else gap)
        if not (raise_next and ch.isalnum()):
            raise_next = False
    return strokes
