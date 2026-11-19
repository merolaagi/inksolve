"""Solving what was read, with the working shown and the answer checked.

  a sum            stepped through in the order of operations: brackets, powers, × and ÷, + and −
  a statement      6 + 1 = 7: both sides worked out, then true or false
  linear equation  expand, gather the unknown on one side, divide
  quadratic        everything to one side, then factorise if it factorises, else the formula
  other equation   SymPy's solver, with numerical answers when there is no exact one
  two unknowns     one equation: make one unknown the subject; several lines: a system
  every answer     substituted back into what was written, to show it works
"""
import sympy as sp

from . import mathparse as MP

L = sp.latex


def _num_steps(tree):
    """Reduce a numeric tree one layer at a time; each layer is a step."""
    steps = []

    def ready(n):
        return n[0] == "num" or (n[0] == "neg" and n[1][0] == "num")

    def val(n):
        return n[1] if n[0] == "num" else -n[1][1]

    def mk(v):
        from fractions import Fraction

        v = Fraction(v)
        txt = str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
        node = ("num", abs(v), txt.lstrip("-"))
        return ("neg", node) if v < 0 else node

    def red(n):
        k = n[0]
        if k in ("num", "var"):
            return n, False
        if k == "paren":
            if ready(n[1]):
                return n[1], True
            c, ch = red(n[1])
            return (c if ready(c) else ("paren", c)), ch
        if k == "neg":
            if n[1][0] == "neg" and ready(n[1][1]):
                return n[1][1], True
            c, ch = red(n[1])
            return ("neg", c), ch
        a, b = n[1], n[2]
        if ready(a) and ready(b):
            x, y = val(a), val(b)
            if k == "add":
                return mk(x + y), True
            if k == "sub":
                return mk(x - y), True
            if k == "mul":
                return mk(x * y), True
            if k in ("div", "frac"):
                if y == 0:
                    raise ZeroDivisionError("You can't divide by zero.")
                return mk(x / y), True
            if k == "pow":
                if y.denominator != 1 or abs(y) > 64:
                    raise ValueError("power too large")
                return mk(x ** int(y)), True
        # powers first, then × ÷, then + −: only reduce the outer op once nothing inside it is left
        ca, cha = red(a)
        cb, chb = red(b)
        out = (k, ca, cb) + tuple(n[3:])
        return out, cha or chb

    cur = tree
    for _ in range(40):
        if ready(cur):
            break
        nxt, ch = red(cur)
        if not ch:
            break
        cur = nxt
        steps.append(MP.to_latex(cur))
    return cur, steps


def _fmt(v):
    v = sp.nsimplify(v) if v.is_number else v
    return L(v)


def _check(eq_tree, sol):
    """Substitute an answer into the statement as written."""
    lhs, rhs = MP.to_sympy(eq_tree[1]), MP.to_sympy(eq_tree[2])
    a, b = sp.simplify(lhs.subs(sol)), sp.simplify(rhs.subs(sol))
    ok = sp.simplify(a - b) == 0
    if not ok:
        try:
            ok = abs(complex(sp.N(a - b))) < 1e-9
        except Exception:
            pass
    subs = ", ".join(f"{L(k)} = {_fmt(v)}" for k, v in sol.items())
    return {"with": subs, "lhs": L(a), "rhs": L(b), "ok": bool(ok)}


def solve_one(tree, mode="solve"):
    out = {"kind": None, "steps": [], "answer": None, "checks": [], "note": None}
    if tree[0] != "eq":
        vs = MP.variables(tree)
        if not vs:
            out["kind"] = "arithmetic"
            try:
                final, steps = _num_steps(tree)
                out["steps"] = [{"latex": s} for s in steps]
                v = MP.to_sympy(tree)
            except ZeroDivisionError as e:
                out["note"] = str(e)
                return out
            except ValueError:
                v = MP.to_sympy(tree)
            out["answer"] = {"latex": _fmt(v), "decimal": None if v.is_integer else str(sp.N(v, 8))}
            return out
        e = MP.to_sympy(tree)
        out["kind"] = "expression"
        forms = []
        for name, f in (("Expanded", sp.expand(e)), ("Factorised", sp.factor(e)), ("Simplified", sp.simplify(e))):
            if f != e and all(L(f) != x["latex"] for x in forms):
                forms.append({"label": name, "latex": L(f)})
        out["steps"] = forms
        out["answer"] = {"latex": L(sp.simplify(e))}
        out["note"] = "There is no equals sign, so there is nothing to solve; here it is rearranged."
        return out

    vs = sorted(MP.variables(tree))
    lhs, rhs = MP.to_sympy(tree[1]), MP.to_sympy(tree[2])
    if not vs:
        out["kind"] = "statement"
        for side, name in ((tree[1], "Left side"), (tree[2], "Right side")):
            try:
                _, st = _num_steps(side)
            except Exception:
                st = []
            out["steps"].append({"label": name, "latex": MP.to_latex(side) + (" = " + st[-1] if st else "")})
        ok = sp.simplify(lhs - rhs) == 0
        out["answer"] = {"latex": (MP.to_latex(tree) + r"\quad\checkmark") if ok else (L(lhs) + r" \neq " + L(rhs)), "true": bool(ok)}
        out["note"] = "True: both sides come to the same number." if ok else "False as written: the two sides come to different numbers."
        return out

    if len(vs) == 1:
        v = MP.SYM[vs[0]]
        expr = sp.expand(lhs - rhs)
        try:
            poly = sp.Poly(expr, v)
            deg = poly.degree()
        except sp.PolynomialError:
            poly, deg = None, None
        if deg == 1:
            out["kind"] = "linear"
            el, er = sp.expand(lhs), sp.expand(rhs)
            if "paren" in repr(tree) or "frac" in repr(tree):
                out["steps"].append({"label": "Expand", "latex": f"{L(el)} = {L(er)}"})
            a = poly.coeff_monomial(v)
            c = -poly.coeff_monomial(1)
            if a < 0:
                a, c = -a, -c
            out["steps"].append({"label": f"Gather {vs[0]} on one side, numbers on the other", "latex": f"{L(a * v)} = {L(c)}"})
            sol = sp.nsimplify(c / a)
            if a != 1:
                out["steps"].append({"label": f"Divide both sides by {L(a)}", "latex": f"{vs[0]} = {L(sol)}"})
            out["answer"] = {"latex": f"{vs[0]} = {L(sol)}", "decimal": None if sol.is_integer else str(sp.N(sol, 8))}
            out["checks"] = [_check(tree, {v: sol})]
            return out
        if deg == 0:
            out["kind"] = "identity" if expr == 0 else "contradiction"
            out["answer"] = {"latex": r"\text{every } " + vs[0] if expr == 0 else r"\text{no solution}"}
            out["note"] = (f"Both sides are the same for every {vs[0]}." if expr == 0 else
                           f"The {vs[0]} cancels and leaves {L(sp.Eq(0, -expr, evaluate=False))}, which is never true.")
            return out
        if deg == 2:
            out["kind"] = "quadratic"
            a, b, c = (poly.coeff_monomial(v ** 2), poly.coeff_monomial(v), poly.coeff_monomial(1))
            out["steps"].append({"label": "Everything on one side", "latex": f"{L(expr)} = 0"})
            fac = sp.factor(expr)
            sols = sp.solve(sp.Eq(expr, 0), v)
            if fac != expr and all(s.is_rational for s in sols):
                out["steps"].append({"label": "Factorise", "latex": f"{L(fac)} = 0"})
                out["steps"].append({"label": "One of the brackets must be zero", "latex": r",\quad ".join(f"{vs[0]} = {L(s)}" for s in sols)})
            else:
                disc = b * b - 4 * a * c
                out["steps"].append({"label": "Quadratic formula", "latex": rf"{vs[0]} = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}},\quad a = {L(a)},\ b = {L(b)},\ c = {L(c)}"})
                out["steps"].append({"label": "Discriminant", "latex": rf"b^2 - 4ac = {L(disc)}"})
                if disc < 0:
                    out["note"] = "The discriminant is negative, so there is no real answer; the complex ones are shown."
            uniq = list(dict.fromkeys(sols))
            out["answer"] = {"latex": r",\quad ".join(f"{vs[0]} = {L(s)}" for s in uniq) or r"\text{no solution}",
                             "decimal": ", ".join(str(sp.N(s, 6)) for s in uniq if not s.is_rational) or None}
            out["checks"] = [_check(tree, {v: s}) for s in uniq]
            return out
        out["kind"] = "equation"
        try:
            sols = sp.solve(sp.Eq(lhs, rhs), v)
        except NotImplementedError:
            sols = []
        if not sols:
            try:
                sols = [sp.nsolve(lhs - rhs, v, g) for g in (-3, 0.5, 3)]
                sols = list(dict.fromkeys(sp.Float(s, 8) for s in sols))
                out["note"] = "No exact answer was found; these are numerical."
            except Exception:
                out["answer"] = {"latex": r"\text{no solution found}"}
                return out
        out["answer"] = {"latex": r",\quad ".join(f"{vs[0]} = {L(s)}" for s in sols)}
        out["checks"] = [_check(tree, {v: s}) for s in sols[:6]]
        return out

    # one equation, two or more unknowns: make one of them the subject
    out["kind"] = "rearrange"
    target = next((x for x in ("y", "z", "b", "a", "x") if x in vs), vs[0])
    t = MP.SYM[target]
    sols = sp.solve(sp.Eq(lhs, rhs), t)
    if not sols:
        out["answer"] = {"latex": r"\text{could not rearrange}"}
        return out
    out["answer"] = {"latex": r",\quad ".join(f"{target} = {L(s)}" for s in sols)}
    out["note"] = f"One equation with {len(vs)} unknowns has no single answer; here it is solved for {target}."
    s0 = sp.expand(sols[0])
    others = [MP.SYM[x] for x in vs if x != target]
    if len(others) == 1 and sp.Poly(s0, others[0]).degree() == 1:
        m, c = sp.Poly(s0, others[0]).coeff_monomial(others[0]), sp.Poly(s0, others[0]).coeff_monomial(1)
        out["steps"].append({"label": "A straight line", "latex": rf"\text{{slope}} = {L(m)},\quad \text{{crosses the {target}-axis at}}\ {L(c)}"})
    return out


def solve_system(trees):
    vs = sorted(set().union(*[MP.variables(t) for t in trees]))
    syms = [MP.SYM[v] for v in vs]
    eqs = [sp.Eq(MP.to_sympy(t[1]), MP.to_sympy(t[2])) for t in trees]
    out = {"kind": "system", "steps": [], "answer": None, "checks": [], "note": None}
    linear = all(sp.Poly(sp.expand(e.lhs - e.rhs), *syms).total_degree() <= 1 for e in eqs)
    if linear and len(syms) == 2 and len(eqs) == 2:
        (a1, b1, c1), (a2, b2, c2) = [[sp.Poly(sp.expand(e.lhs - e.rhs), *syms).coeff_monomial(m) for m in (syms[0], syms[1], 1)] for e in eqs]
        out["steps"].append({"label": "Line them up", "latex": rf"\begin{{aligned}} {L(a1 * syms[0] + b1 * syms[1])} &= {L(-c1)} \\ {L(a2 * syms[0] + b2 * syms[1])} &= {L(-c2)} \end{{aligned}}"})
        if b1 != 0 and b2 != 0:
            out["steps"].append({"label": f"Multiply to match the {vs[1]} terms, then subtract to remove {vs[1]}",
                                 "latex": rf"{L((a1 * b2 - a2 * b1) * syms[0])} = {L(-c1 * b2 + c2 * b1)}"})
    try:
        sol = sp.solve(eqs, syms, dict=True)
    except Exception:
        sol = []
    if not sol:
        out["answer"] = {"latex": r"\text{no solution}"}
        out["note"] = "These equations can't all be true at once."
        return out
    s = sol[0]
    if len(s) < len(syms):
        out["note"] = "There are infinitely many answers; here they are in terms of the free unknown."
    out["answer"] = {"latex": r",\quad ".join(f"{L(k)} = {L(v)}" for k, v in s.items())}
    out["checks"] = [_check(t, s) for t in trees] if len(s) == len(syms) else []
    return out


def solve(trees, mode="solve"):
    """trees: one per line. Lines that are all equations with unknowns, two or more of them, form a system."""
    eqs = [t for t in trees if t[0] == "eq" and MP.variables(t)]
    if len(eqs) >= 2 and len(eqs) == len(trees):
        return [solve_system(trees)]
    return [solve_one(t, mode) for t in trees]
