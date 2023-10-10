from complex import Complex
from utils import *

from functools import reduce
import logging

import z3

logger = logging.getLogger("check")
def check_barrier(barrier : sym.Poly,
                  constraints : Dict[str, List[sym.Poly]],
                  Z : List[sym.Symbol] = [],
                  unitary = np.eye(2,2),
                  k = 1,
                  eps = 0.01,
                  log_level = logging.INFO):
    logger.setLevel(log_level)
    d = calculate_d(k, eps)
    unitary_k = generate_unitary_k(k, unitary)
    variables = Z + [z.conjugate() for z in Z]
    var_z3_dict = dict(zip(Z, [Complex(var.name) for var in Z]))
    
    z3_barrier = _sympy_poly_to_z3(var_z3_dict, barrier)
    z3_diff = _sympy_poly_to_z3(var_z3_dict, sym.poly(barrier.subs(zip(Z, np.dot(unitary, Z))) - barrier, variables, domain=sym.CC))
    z3_k_diff = _sympy_poly_to_z3(var_z3_dict, sym.poly(barrier.subs(zip(Z, np.dot(unitary_k, Z))) - barrier, variables, domain=sym.CC))

    for key in constraints: constraints[key] = [_sympy_poly_to_z3(var_z3_dict, p).r >= 0 for p in constraints[key]]

    def _check(s : z3.Solver, cond):
        s.push()
        s.add(cond)
        logger.debug(str(s))
        sat = s.check()
        if sat == z3.unsat: logger.info("Constraint satisfied.")
        elif sat == z3.unknown: logger.warning("Solver returned unkown. Function may not satisfy barrier certificate constraint.")
        elif sat == z3.sat:
            m = s.model()
            s2 = z3.Solver()
            s2.add(Complex('barrier') == z3_barrier)
            for v in m: s2.add(v() == m[v()])
            s2.check()
            logger.error("Counter example: " + str(s2.model()))
            exit()
        s.pop()
    
    s = z3.Solver()
    logger.info("Barrier real")
    _check(s, (z3.And(constraints[INVARIANT]), z3.Not(z3.And(z3_barrier.i >= -1e-10, z3_barrier.i <= 1e-10))))
    logger.info("Check " + INIT)
    _check(s, (z3.And(constraints[INIT]), z3_barrier.r > 0))
    logger.info("Check " + UNSAFE)
    _check(s, (z3.And(constraints[UNSAFE]), z3_barrier.r < d))
    logger.info("Check " + DIFF)
    _check(s, (z3.And(constraints[INVARIANT]), z3_diff.r > eps))
    logger.info("Check " + INDUCTIVE)
    _check(s, (z3.And(constraints[INVARIANT]), z3_k_diff.r > 0))
    logger.info("All constraints checked.")

# Based on: https://stackoverflow.com/a/38980538/19768075
def _sympy_poly_to_z3(var_map, e) -> z3.ExprRef:
    rv = None
    if isinstance(e, sym.Poly): return _sympy_poly_to_z3(var_map, e.as_expr())
    elif not isinstance(e, sym.Expr): raise RuntimeError("Expected sympy Expr: " + repr(e))
    if isinstance(e, sym.Symbol): rv = var_map[e]
    elif isinstance(e, sym.Number): rv = float(e)
    elif isinstance(e, sym.Mul): rv = reduce((lambda x, y: x * y), [_sympy_poly_to_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Add): rv = sum([_sympy_poly_to_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Pow): rv = _sympy_poly_to_z3(var_map, e.args[0]) ** _sympy_poly_to_z3(var_map, e.args[1])
    elif isinstance(e, sym.conjugate): rv = _sympy_poly_to_z3(var_map, e.args[0]).conj()
    if rv is None: raise Exception(repr(e), type(e))
    return rv