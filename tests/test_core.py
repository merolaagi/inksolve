"""Core checks: python -m pytest tests (or python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import decode as D, mathparse as MP, segment as S, solve as SV, synth  # noqa: E402
from server.glyphs import CLASSES  # noqa: E402


def solve_text(s):
    return SV.solve([MP.parse(MP.tokenize_text(p)) for p in s.split(";")])


def test_parse_and_grammar():
    assert MP.valid(MP.tokenize_text("2x+3=11"))
    assert not MP.valid(["x", "3"])
    assert not MP.valid(["6", "+", "=", "7"])
    assert MP.to_text(MP.parse(MP.tokenize_text("x^2-5x+6=0"))) == "x^2 - 5x + 6 = 0"


def test_solver():
    assert solve_text("2x+3=11")[0]["answer"]["latex"] == "x = 4"
    assert solve_text("x^2-5x+6=0")[0]["answer"]["latex"] == r"x = 2,\quad x = 3"
    assert solve_text("x+y=10;x-y=2")[0]["answer"]["latex"] == r"x = 6,\quad y = 4"
    assert solve_text("6+1=1")[0]["answer"]["true"] is False
    assert solve_text("(3+5)*2")[0]["answer"]["latex"] == "16"
    assert all(c["ok"] for c in solve_text("2(x+1)=3x-4")[0]["checks"])


def fake_probs(seq, unsure=None):
    """Confident probabilities for each symbol, except marks given as {i: {class: p}}."""
    P = np.full((len(seq), len(CLASSES)), 1e-4, np.float32)
    for i, c in enumerate(seq):
        P[i, CLASSES.index(c)] = 1.0
    for i, d in (unsure or {}).items():
        P[i] = 1e-4
        for c, p in d.items():
            P[i, CLASSES.index(c)] = p
    return P / P.sum(1, keepdims=True)


def stream(n):
    return [("glyph", {"i": i}) for i in range(n)]


def test_context_reads_7_but_grading_does_not():
    P = fake_probs("6+1=1", {4: {"1": 0.62, "7": 0.37}})
    P[3] = 0
    P[3, CLASSES.index("=")] = 1
    solve = D.decode_line(stream(5), P, "solve")[0]
    grade = D.decode_line(stream(5), P, "grade")[0]
    assert solve["h"].tokens[-1] == "7"
    assert grade["h"].tokens[-1] == "1"


def test_clear_1_is_not_overruled():
    P = fake_probs("6+1=1", {4: {"1": 0.995, "7": 0.005}})
    assert D.decode_line(stream(5), P, "solve")[0]["h"].tokens[-1] == "1"


def test_x_is_letter_or_times_by_context():
    assert D.decode_line(stream(5), fake_probs("3x4=1"), "grade")[0]["h"].tokens[1] == "*"
    assert D.decode_line(stream(4), fake_probs("2x+3"), "grade")[0]["h"].tokens[1] == "x"


def test_segmentation_on_synthetic_ink():
    ok = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        lay = S.segment(S.pieces_from_strokes(synth.expression_strokes("x^2-5x+6=0", rng)))
        line = lay["lines"]
        ok += len(line) == 1 and len(lay["glyphs"]) == 9 and bool(line[0][0]["sup"])
    assert ok >= 17


def test_inside_lab_step_is_sgd():
    """One step in the lab moves each weight by -lr x (clipped) gradient, and only weights with a gradient move."""
    import base64

    import torch

    from server import inside as IN, model as M

    class Reg:
        def load(self, mid=None):
            torch.manual_seed(0)
            n = M.Net()
            n.eval()
            return n, 1.0, {"name": "test"}
    lab = IN.Lab(Reg())
    lab.reset("trained", "sgd", 0.01)
    x = torch.tensor(np.asarray(synth.sample("7", np.random.default_rng(1)), np.float32))[None, None]
    out = lab.snapshot(x, label="7", step=True)
    dec = lambda b: np.frombuffer(base64.b64decode(b), np.float16).astype(np.float32)
    scale = min(1.0, IN.CLIP / out["grad_norm"])
    for l in out["layers"]:
        g, d = dec(l["g"]), dec(l["d"])
        assert np.allclose(d, -0.01 * scale * g, atol=2e-4), l["name"]
        assert ((g == 0) <= (np.abs(d) < 1e-6)).all()
    assert out["counts"]["total"] == sum(p.numel() for p in lab.net.parameters())


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
