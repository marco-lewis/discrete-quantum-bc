from src.complex import Complex
from src.utils import *

from functools import reduce
import logging

import z3

logger = logging.getLogger("check")

def raise_error(msg):
    logger.error(msg)
    exit()

# TODO: Change so unitaries and barriers are separate
def check_barrier(barriers : list[tuple[np.ndarray, sym.Poly]],
                  g : dict[str, list[sym.Poly]],
                  Z : list[sym.Symbol] = [],
                  idx_pairs : list[tuple[int,int]] = [()],
                  chunks : list[tuple[np.ndarray,int,int]] = [],
                  k = 1,
                  eps = 0.01,
                  gamma = 0.01,
                  log_level = logging.INFO):
    logger.setLevel(log_level)
    d = calculate_d(k, eps, gamma)
    variables = Z + [z.conjugate() for z in Z]
    var_z3_dict = dict(zip(Z, [Complex(var.name) for var in Z]))
    
    # Barriers
    z3_barriers = [(unitary, _sympy_poly_to_z3(var_z3_dict, barrier)) for unitary, barrier in barriers]
    # Difference
    z3_diffs = [_sympy_poly_to_z3(var_z3_dict, sym.poly(barrier.subs(zip(Z, np.dot(unitary, Z))) - barrier, variables, domain=sym.CC)) for unitary, barrier in barriers]
    # Change
    z3_changes = [_sympy_poly_to_z3(var_z3_dict, sym.poly(barriers[i2][1] - barriers[i1][1], variables, domain=sym.CC)) for i1, i2 in idx_pairs]
    # Inductive
    z3_k_diffs = [_sympy_poly_to_z3(var_z3_dict, sym.poly(barriers[i2][1].subs(zip(Z, np.dot(unitary_k, Z))) - barriers[i1][1], variables, domain=sym.CC)) for unitary_k, i1, i2 in chunks]

    z3_constraints : dict[str, z3.ExprRef] = {}
    for key in g: z3_constraints[key] = [_sympy_poly_to_z3(var_z3_dict, p).r >= 0 for p in g[key]]

    def _check(s : z3.Solver, conds : list[z3.ExprRef]):
        s.push()
        [s.add(z3.simplify(cond)) for cond in conds]
        logger.debug("Conditions in solver:\n" + str(s))
        sat = s.check()
        if sat == z3.unsat: logger.info("Constraint satisfied.")
        elif sat == z3.unknown: logger.warning("Solver returned unkown. Function may not satisfy barrier certificate constraint.")
        elif sat == z3.sat:
            m = s.model()
            s2 = z3.Solver()
            s2.add([Complex('barrier' + str(i)) == z3_barriers[i] for i in range(len(z3_barriers))])
            for v in m: s2.add(v() == m[v()])
            s2.check()
            raise_error("Counter example: " + str(s2.model()))
        s.pop()
    
    tactic = z3.Then('solve-eqs','smt')
    s = tactic.solver()
    for unitary, z3_barrier in z3_barriers:
        logger.info("Unitary\n" + str(unitary))
        logger.info("Check barrier real")
        _check(s, [z3.And(z3_constraints[INVARIANT]), z3.Not(z3_barrier.i == 0)])
        logger.info("Check " + INIT)
        _check(s, [z3.And(z3_constraints[INIT]), z3_barrier.r > 0])
        logger.info("Check " + UNSAFE)
        _check(s, [z3.And(z3_constraints[UNSAFE]), z3_barrier.r < d])
    for z3_diff in z3_diffs:
        logger.info("Check " + DIFF + " " + z3_diffs.index(z3_diff))
        _check(s, [z3.And(z3_constraints[INVARIANT]), z3_diff.r > eps])
    for z3_change in z3_changes:
        logger.info("Check " + CHANGE + " " + z3_changes.index(z3_change))
        _check(s, [z3.And(z3_constraints[INVARIANT]), z3_change.r > gamma])
    for z3_k_diff in z3_k_diffs:
        logger.info("Check " + INDUCTIVE + " " + z3_k_diffs.index(z3_k_diff))
        _check(s, [z3.And(z3_constraints[INVARIANT]), z3_k_diff.r > 0])
    logger.info("All constraints checked.")

# Based on: https://stackoverflow.com/a/38980538/19768075
def _sympy_poly_to_z3(var_map, e) -> z3.ExprRef:
    rv = None
    if isinstance(e, sym.Poly): return _sympy_poly_to_z3(var_map, e.as_expr())
    elif not isinstance(e, sym.Expr): raise_error("Expected sympy Expr: " + repr(e))
    if isinstance(e, sym.Symbol): rv = var_map[e]
    elif isinstance(e, sym.Number): rv = float(e)
    elif isinstance(e, sym.Mul): rv = reduce((lambda x, y: x * y), [_sympy_poly_to_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Add): rv = sum([_sympy_poly_to_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Pow): rv = _sympy_poly_to_z3(var_map, e.args[0]) ** _sympy_poly_to_z3(var_map, e.args[1])
    elif isinstance(e, sym.conjugate): rv = _sympy_poly_to_z3(var_map, e.args[0]).conj()
    if rv is None: raise_error("Unable to handle input " + repr(e) + " (" + str(type(e)) + ")")
    return rv