from complex import Complex
from utils import *

from collections import defaultdict
from functools import reduce

import z3

# Need to convert a generated sympy barrier into Z3 expressions

# Then check against some conditions given in Z3
def check_barrier(barrier : sym.Poly, constraints : Dict[str, List[sym.Poly]], Z=[], unitary=np.eye(2,2)):
    variables = Z + [z.conjugate() for z in Z]
    var_z3_dict = dict(zip(Z, [Complex(var.name) for var in Z]))
    z3_vars = list(var_z3_dict.values())
    z3_barrier = _sympy_poly_2_z3(var_z3_dict, barrier)
    z3_diff = _sympy_poly_2_z3(var_z3_dict, sym.poly(barrier.subs(zip(Z, np.dot(unitary, Z))) - barrier, variables, domain=sym.CC))
    for key in constraints: constraints[key] = [_sympy_poly_2_z3(var_z3_dict, p).r >= 0 for p in constraints[key]]

    def _check(s : z3.Solver, cond):
        s.push()
        s.add(cond)
        sat = s.check()
        print(sat)
        if not(sat == z3.unsat):
            raise Exception("Counter example: " + str(s.model()))
        s.pop()
    s = z3.Solver()
    print("Barrier real")
    _check(s, (sum([v * v.conj() for v in z3_vars]) == 1, z3.Not(z3_barrier.i == 0)))
    print(INIT)
    _check(s, (z3.And(constraints[INIT]), z3_barrier.r > 0))
    print(UNSAFE)
    _check(s, (z3.And(constraints[UNSAFE]), z3_barrier.r < 0))
    print(INVARIANT)
    _check(s, (z3.And(constraints[INVARIANT], z3_diff > 0)))

# Based on: https://stackoverflow.com/a/38980538/19768075
def _sympy_poly_2_z3(var_map, e) -> z3.ExprRef:
    rv = None
    if isinstance(e, sym.Poly): return _sympy_poly_2_z3(var_map, e.as_expr())
    elif not isinstance(e, sym.Expr): raise RuntimeError("Expected sympy Expr: " + repr(e))
    if isinstance(e, sym.Symbol): rv = var_map[e]
    elif isinstance(e, sym.Number): rv = float(e)
    elif isinstance(e, sym.Mul): rv = reduce((lambda x, y: x * y), [_sympy_poly_2_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Add): rv = sum([_sympy_poly_2_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Pow): rv = _sympy_poly_2_z3(var_map, e.args[0]) ** _sympy_poly_2_z3(var_map, e.args[1])
    elif isinstance(e, sym.conjugate): rv = _sympy_poly_2_z3(var_map, e.args[0]).conj()
    if rv is None: raise Exception(repr(e), type(e))
    return rv