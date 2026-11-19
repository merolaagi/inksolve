"""The language of what can be written: tokens, a grammar, and a parser.

Tokens are single symbols plus a few structural ones the layout adds:
  "^(" ... ")^"            an exponent group (things written raised after a base)
  "[/" num "/|/" den "/]"  a written fraction (symbols above and below a bar)
  "."                      a decimal point

The parser turns tokens into a small tree that keeps exactly what was written (so the
reading is shown as written, and arithmetic can be stepped through in order), and
converts that tree to SymPy for solving and to LaTeX for display.

`Prefix` is the incremental check the decoder uses: given the tokens so far, may this
token come next? It is what lets "6 + 1 = 7" and "2x + 3" be read while "6 + = 7" or
"x3" are never considered.
"""
from fractions import Fraction

import sympy as sp

VARS = set("xyabz")
BINOPS = {"+", "-", "*", "/", "÷"}
OPEN_GROUPS = {"^(": ")^", "[/": "/|/", "/|/": "/]"}
CLOSERS = {")^", "/|/", "/]"}


def kind(t):
    if t.isdigit():
        return "D"
    if t in VARS:
        return "V"
    if t == ".":
        return "P"
    if t in BINOPS:
        return "O"
    if t == "=":
        return "E"
    if t == "(":
        return "L"
    if t == ")":
        return "R"
    if t in ("^(", "[/"):
        return "G"            # opens a group
    if t in CLOSERS:
        return "C"
    raise ValueError(t)


# ------------------------------------------------------------------ incremental validity
class Prefix:
    """Immutable-ish state for "is this a valid start of a statement?"."""
    __slots__ = ("last", "stack", "eq", "point", "numlen", "lead0")

    def __init__(self):
        self.last = "S"            # S start, D, V, P, O, U (unary minus), E, L, R, C (closed group)
        self.stack = []            # open groups and parentheses: "(", "^(", "[/", "/|/"
        self.eq = 0
        self.point = False         # the number being written already has a decimal point
        self.numlen = 0
        self.lead0 = False

    def copy(self):
        p = Prefix.__new__(Prefix)
        p.last, p.stack, p.eq, p.point, p.numlen, p.lead0 = self.last, list(self.stack), self.eq, self.point, self.numlen, self.lead0
        return p

    def _starts_operand(self):
        return self.last in ("S", "O", "U", "E", "L", "G")

    def _ends_operand(self):
        return self.last in ("D", "V", "R", "C")

    def push(self, t):
        """Returns (ok, penalty). Penalties are soft dislikes (a number written 07), not errors."""
        k = kind(t)
        pen = 0.0
        if k == "D":
            if self.last == "V" or self.last == "R" or self.last == "C":
                return False, 0
            if self.last == "D":
                if self.lead0 and not self.point:
                    pen -= 2.5        # a number that starts 0 and goes on: 07
                self.numlen += 1
            elif self.last == "P":
                pass
            else:
                self.point, self.numlen, self.lead0 = False, 1, t == "0"
            self.last = "D"
            return True, pen
        if k == "P":
            if self.last != "D" or self.point:
                return False, 0
            self.point, self.last, self.lead0 = True, "P", False
            return True, 0
        if k == "V":
            if self.last == "P":
                return False, 0
            if self.last == "V":
                pen -= 0.3            # xy is fine, but two letters in a row is a little less likely
            self.last = "V"
            return True, pen
        if k == "O":
            if t == "-" and self._starts_operand() and self.last != "U":
                pen = -0.8 if self.last == "O" else 0.0       # 2 × -3 is allowed but unusual
                self.last = "U"
                return True, pen
            if not self._ends_operand():
                return False, 0
            self.last = "O"
            return True, 0
        if k == "E":
            if not self._ends_operand() or self.stack or self.eq >= 1:
                return False, 0
            self.eq += 1
            self.last = "E"
            return True, 0
        if k == "L":
            if self.last == "P":
                return False, 0
            self.stack.append("(")
            self.last = "L"
            return True, 0
        if k == "R":
            if not self._ends_operand() or not self.stack or self.stack[-1] != "(":
                return False, 0
            self.stack.pop()
            self.last = "R"
            return True, 0
        if k == "G":
            if t == "^(" and not self._ends_operand():
                return False, 0
            if t == "[/" and self.last == "P":
                return False, 0
            self.stack.append(t)
            self.last = "G"
            return True, 0
        if k == "C":
            want = {")^": "^(", "/|/": "[/", "/]": "/|/"}[t]
            if not self._ends_operand() or not self.stack or self.stack[-1] != want:
                return False, 0
            self.stack.pop()
            if t == "/|/":
                self.stack.append("/|/")
                self.last = "G"
            else:
                self.last = "C"
            return True, 0
        return False, 0

    def complete(self):
        return self._ends_operand() and not self.stack


def valid(tokens):
    p = Prefix()
    for t in tokens:
        ok, _ = p.push(t)
        if not ok:
            return False
    return p.complete()


# ------------------------------------------------------------------ parser -> tree
class ParseError(ValueError):
    pass


def parse(tokens):
    """tokens -> ("eq", lhs, rhs) or an expression tree. Trees are tuples:
    ("num", Fraction, text) ("var", name) ("add"|"sub"|"mul"|"div"|"pow", a, b, style) ("neg", a) ("paren", a) ("frac", a, b)."""
    toks = list(tokens)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take(expect=None):
        t = peek()
        if t is None or (expect is not None and t != expect):
            raise ParseError(f"expected {expect or 'more'} at {pos[0]}")
        pos[0] += 1
        return t

    def expr():
        a = term()
        while peek() in ("+", "-"):
            op = take()
            b = term()
            a = ("add" if op == "+" else "sub", a, b, None)
        return a

    def term():
        a = unary()
        while True:
            t = peek()
            if t in ("*", "/", "÷"):
                take()
                b = unary()
                a = ("mul", a, b, "times") if t == "*" else ("div", a, b, "obelus" if t == "÷" else "slash")
            elif t is not None and (t in VARS or t == "(" or t == "[/" or (t.isdigit() and a[0] in ("paren",))):
                b = power()
                a = ("mul", a, b, "implicit")
            else:
                return a

    def unary():
        if peek() == "-":
            take()
            return ("neg", unary())
        if peek() == "+":
            take()
            return unary()
        return power()

    def power():
        a = atom()
        if peek() == "^(":
            take()
            e = expr()
            take(")^")
            a = ("pow", a, e, None)
        return a

    def atom():
        t = peek()
        if t is None:
            raise ParseError("ends too soon")
        if t.isdigit():
            s = ""
            while peek() is not None and (peek().isdigit() or peek() == "."):
                s += take()
            return ("num", Fraction(s), s)
        if t in VARS:
            take()
            return ("var", t)
        if t == "(":
            take()
            e = expr()
            take(")")
            return ("paren", e)
        if t == "[/":
            take()
            n = expr()
            take("/|/")
            d = expr()
            take("/]")
            return ("frac", n, d)
        raise ParseError(f"unexpected {t}")

    lhs = expr()
    if peek() == "=":
        take()
        rhs = expr()
        out = ("eq", lhs, rhs)
    else:
        out = lhs
    if peek() is not None:
        raise ParseError(f"unexpected {peek()} at {pos[0]}")
    return out


# ------------------------------------------------------------------ tree -> SymPy / LaTeX / text
SYM = {v: sp.Symbol(v) for v in VARS}


def to_sympy(n):
    k = n[0]
    if k == "eq":
        return sp.Eq(to_sympy(n[1]), to_sympy(n[2]), evaluate=False)
    if k == "num":
        return sp.Rational(n[1].numerator, n[1].denominator)
    if k == "var":
        return SYM[n[1]]
    if k == "neg":
        return -to_sympy(n[1])
    if k == "paren":
        return to_sympy(n[1])
    a, b = to_sympy(n[1]), to_sympy(n[2])
    if k == "add":
        return a + b
    if k == "sub":
        return a - b
    if k == "mul":
        return a * b
    if k in ("div", "frac"):
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b
    if k == "pow":
        return a ** b
    raise ValueError(k)


PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 3, "pow": 4}


def to_latex(n):
    k = n[0]
    if k == "eq":
        return f"{to_latex(n[1])} = {to_latex(n[2])}"
    if k == "num":
        return n[2]
    if k == "var":
        return n[1]
    if k == "neg":
        return "-" + to_latex(n[1])
    if k == "paren":
        return r"\left(" + to_latex(n[1]) + r"\right)"
    if k == "frac":
        return r"\frac{" + to_latex(n[1]) + "}{" + to_latex(n[2]) + "}"
    a, b = to_latex(n[1]), to_latex(n[2])
    if k == "add":
        return f"{a} + {b}"
    if k == "sub":
        return f"{a} - {b}"
    if k == "mul":
        if n[3] == "implicit":
            return f"{a}{b}"
        return rf"{a} \times {b}"
    if k == "div":
        return rf"{a} \div {b}" if n[3] == "obelus" else f"{a} / {b}"
    if k == "pow":
        base = a if n[1][0] in ("num", "var", "paren") else r"\left(" + a + r"\right)"
        return base + "^{" + b + "}"
    raise ValueError(k)


def to_text(n):
    k = n[0]
    if k == "eq":
        return f"{to_text(n[1])} = {to_text(n[2])}"
    if k == "num":
        return n[2]
    if k == "var":
        return n[1]
    if k == "neg":
        return "-" + to_text(n[1])
    if k == "paren":
        return "(" + to_text(n[1]) + ")"
    if k == "frac":
        return "(" + to_text(n[1]) + ")/(" + to_text(n[2]) + ")"
    a, b = to_text(n[1]), to_text(n[2])
    sym = {"add": " + ", "sub": " - ", "pow": "^"}.get(k)
    if k == "mul":
        sym = "" if n[3] == "implicit" else " × "
    if k == "div":
        sym = " ÷ " if n[3] == "obelus" else " / "
    if k == "pow" and not (b.isdigit() or b in VARS):
        b = "(" + b + ")"
    return f"{a}{sym}{b}"


def variables(n):
    if n[0] == "var":
        return {n[1]}
    out = set()
    for c in n[1:]:
        if isinstance(c, tuple):
            out |= variables(c)
    return out


# ------------------------------------------------------------------ typed input -> tokens
def tokenize_text(s):
    """Typed maths -> tokens. Accepts 2x+3=11, 3*4, 12÷4, x^2, x^(n+1), x², (1/2)... and 'times'."""
    s = s.strip().replace("×", "*").replace("·", "*").replace("−", "-").replace("–", "-").replace(":", "÷")
    sup = {"²": "2", "³": "3", "⁴": "4", "⁰": "0", "¹": "1", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in sup:
            out += ["^(", sup[c], ")^"]
            i += 1
            continue
        if c == "^" or (c == "*" and s[i:i + 2] == "**"):
            i += 2 if c == "*" else 1
            while i < len(s) and s[i].isspace():
                i += 1
            if i < len(s) and s[i] == "(":
                depth, j = 0, i
                while j < len(s):
                    depth += s[j] == "("
                    depth -= s[j] == ")"
                    if depth == 0:
                        break
                    j += 1
                inner = tokenize_text(s[i + 1:j])
                out += ["^("] + inner + [")^"]
                i = j + 1
            else:
                j = i
                if j < len(s) and s[j] == "-":
                    j += 1
                while j < len(s) and (s[j].isdigit() or s[j] == "."):
                    j += 1
                if j == i and i < len(s) and s[i].lower() in VARS:
                    j = i + 1
                out += ["^("] + tokenize_text(s[i:j]) + [")^"]
                i = j
            continue
        if c.isdigit() or c == ".":
            out.append(c)
            i += 1
            continue
        if c.lower() in VARS:
            out.append(c.lower())
            i += 1
            continue
        if c in "+-*/÷=()":
            out.append(c)
            i += 1
            continue
        raise ParseError(f"InkSolve doesn't read {c!r} yet.")
    return out
