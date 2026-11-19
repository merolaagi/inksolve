"""Glyphs: the symbols InkSolve reads, and how any mark becomes the same 28x28 image.

Every source (pen strokes from the canvas, a blob cut from a photo, an MNIST digit,
a HASY symbol, a synthetic stroke) passes through one normalisation, the way MNIST
was made: crop to the ink, scale the longest side to 20 pixels, centre by the ink's
centre of mass in a 28x28 square. Strokes are re-drawn at a fixed pen width after
scaling, so a small glyph and a large one look alike to the model.

'x' is one glyph class for both the letter x and the times sign: on paper they are
the same mark, and only context (what is around it) can tell them apart.
"""
import io
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "+", "-", "=", "x", "y", "a", "b", "z", "(", ")", "÷", "/"]
IDX = {c: i for i, c in enumerate(CLASSES)}
DIGITS = CLASSES[:10]
SIZE, BOX = 28, 20
# pairs that are easy to confuse; reported separately everywhere
PAIRS = [("1", "7"), ("b", "6"), ("z", "2"), ("x", "+"), ("5", "3"), ("4", "9"), ("(", "1"), ("/", "1")]


def _center(img):
    """Paste a glyph (<= 20x20) into 28x28 so its centre of mass sits at the centre."""
    out = np.zeros((SIZE, SIZE), np.float32)
    h, w = img.shape
    if img.sum() <= 0:
        return out
    cy, cx = ndimage.center_of_mass(img)
    oy, ox = int(round(SIZE / 2 - cy)), int(round(SIZE / 2 - cx))
    oy, ox = max(0, min(SIZE - h, oy)), max(0, min(SIZE - w, ox))
    out[oy:oy + h, ox:ox + w] = img
    return out


def render_strokes(strokes, width=2.2, supersample=4):
    """Pen strokes [[x, y], ...] (any coordinates) -> 28x28 float in [0, 1], ink = 1."""
    pts = [np.asarray(s, np.float32).reshape(-1, 2) for s in strokes if len(s)]
    if not pts:
        return np.zeros((SIZE, SIZE), np.float32)
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    span = max(float((hi - lo).max()), 1e-6)
    scale = BOX / span
    w = int(math.ceil((hi[0] - lo[0]) * scale)) + 1
    h = int(math.ceil((hi[1] - lo[1]) * scale)) + 1
    w, h = max(1, min(BOX + 1, w)), max(1, min(BOX + 1, h))
    pad = width
    S = supersample
    W, H = int((w + 2 * pad) * S), int((h + 2 * pad) * S)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    lw = max(1, int(round(width * S)))
    for p in pts:
        q = [((x - lo[0]) * scale * S + pad * S, (y - lo[1]) * scale * S + pad * S) for x, y in p]
        if len(q) == 1:
            x, y = q[0]
            d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill=255)
        else:
            d.line(q, fill=255, width=lw, joint="curve")
            for x, y in (q[0], q[-1]):
                d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill=255)
    small = im.resize((max(1, W // S), max(1, H // S)), Image.LANCZOS)
    a = np.asarray(small, np.float32) / 255.0
    return _fit(a)


def _fit(a):
    """Crop to ink, scale the longest side to BOX, centre in 28x28."""
    ys, xs = np.nonzero(a > 0.05)
    if len(ys) == 0:
        return np.zeros((SIZE, SIZE), np.float32)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = BOX / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS if s < 1 else Image.BILINEAR)
    return _center(np.asarray(im, np.float32) / 255.0)


def normalize_mask(mask, target_width=2.2):
    """A binary or grey ink mask (ink high) cut from a photo or dataset -> 28x28.
    Stroke width is brought towards the pen width the strokes are drawn with."""
    a = np.asarray(mask, np.float32)
    if a.max() > 1:
        a = a / 255.0
    ys, xs = np.nonzero(a > 0.2)
    if len(ys) == 0:
        return np.zeros((SIZE, SIZE), np.float32)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # estimate stroke width in the scaled glyph: ink area / skeleton length ~ width
    h, w = a.shape
    s = BOX / max(h, w)
    b = a > 0.4
    area = b.sum()
    edges = b & ~ndimage.binary_erosion(b)
    length = max(1.0, edges.sum() / 2.0)
    width_scaled = (area / length) * s
    out = _fit(a)
    if width_scaled < target_width * 0.6:
        out = np.maximum(out, ndimage.grey_dilation(out, size=(2, 2)))
    elif width_scaled > target_width * 2.2:
        out = ndimage.grey_erosion(out, size=(2, 2))
        out = _center_again(out)
    return np.clip(out, 0, 1)


def _center_again(a):
    return _fit(a)


# ------------------------------------------------------------------ augmentation (training only)
def augment(img, rng, strength=1.0):
    """Random small affine, elastic wobble, pen width change and noise; keeps the glyph recognisable."""
    a = img
    ang = rng.normal(0, 8 * strength) * math.pi / 180
    shear = rng.normal(0, 0.15 * strength)
    sc = 1 + rng.normal(0, 0.07 * strength)
    asp = 1 + rng.normal(0, 0.07 * strength)
    c, s_ = math.cos(ang), math.sin(ang)
    M = np.array([[c, -s_], [s_, c]]) @ np.array([[1, shear], [0, 1]]) @ np.diag([sc * asp, sc / asp])
    Minv = np.linalg.inv(M)
    ctr = np.array([SIZE / 2, SIZE / 2])
    off = ctr - Minv @ ctr + rng.normal(0, 1.0 * strength, 2)
    a = ndimage.affine_transform(a, Minv, offset=off, order=1, mode="constant")
    if strength > 0 and rng.random() < 0.5:                      # elastic wobble
        dx = ndimage.gaussian_filter(rng.normal(0, 1, a.shape), 3) * 2.2 * strength
        dy = ndimage.gaussian_filter(rng.normal(0, 1, a.shape), 3) * 2.2 * strength
        yy, xx = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
        a = ndimage.map_coordinates(a, [yy + dy, xx + dx], order=1, mode="constant")
    r = rng.random()
    if r < 0.2:
        a = ndimage.grey_dilation(a, size=(2, 2))
    elif r < 0.35:
        a = ndimage.grey_erosion(a, size=(2, 1)) if rng.random() < .5 else ndimage.grey_erosion(a, size=(1, 2))
    if rng.random() < 0.3:
        a = a + rng.normal(0, 0.05, a.shape)
    return np.clip(a, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ images for the page
def to_png(a, scale=2, invert=True):
    a = np.asarray(a, np.float32)
    if a.max() > 1:
        a = a / 255.0
    im = Image.fromarray(np.clip((1 - a if invert else a) * 255, 0, 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def to_data_url(a, scale=2):
    import base64

    return "data:image/png;base64," + base64.b64encode(to_png(a, scale)).decode()
