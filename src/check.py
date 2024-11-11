from src.complex import Complex, I
from src.typings import *
from src.utils import *
from src.z3todreal import run_dreal

from functools import reduce
import logging
import sys

import z3

logger = logging.getLogger("check")

def raise_error(msg):
    logger.error(msg)
    sys.exit(1)

def check_barrier(barrier_certificate : BarrierCertificate,
                  g : SemiAlgebraicDict,
                  Z : list[sym.Symbol] = [],
                  idx_pairs : list[tuple[int,int]] = [()],
                  chunks : list[Chunk] = [],
                  k = 1,
                  eps = 0.01,
                  gamma = 0.01,
                  timeout = 300,
                  log_level = logging.INFO):
    logger.setLevel(log_level)
    
    variables = generate_variables(Z)
    var_z3_dict = dict(zip(Z, [Complex(var.name) for var in Z]))
    
    logger.info("Checking all barriers are real.")
    for unitary, barrier in barrier_certificate:
        # 2i * im(barrier) barrier
        barrier_imag = round_expr(sym.simplify(barrier - sym.conjugate(barrier)), num_digits=15)
        if not barrier_imag == 0:
            logger.error(barrier_imag)
            raise_error("The following barrier is not real.\n" + str(unitary) + ": " + str(barrier))
    logger.info("All barriers real.")

    # Barriers
    z3_barriers = [(unitary, _sympy_poly_to_z3(var_z3_dict, barrier)) for unitary, barrier in barrier_certificate]
    # Difference
    z3_diffs = [_sympy_poly_to_z3(var_z3_dict, sym.poly(barrier.as_expr().subs(zip(Z, np.dot(unitary, Z)), simultaneous=True) - barrier, variables, domain=sym.CC)) for unitary, barrier in barrier_certificate] if k > 1 else []
    # Change
    z3_changes = [_sympy_poly_to_z3(var_z3_dict, sym.poly(barrier_certificate[i2][1] - barrier_certificate[i1][1], variables, domain=sym.CC)) for i1, i2 in idx_pairs]
    # Inductive
    z3_k_diffs = [_sympy_poly_to_z3(var_z3_dict, sym.poly(barrier_certificate[i2][1].as_expr().subs(zip(Z, np.dot(unitary_k, Z)), simultaneous=True) - barrier_certificate[i1][1], variables, domain=sym.CC)) for unitary_k, i1, i2 in chunks]

    z3_constraints : dict[str, z3.ExprRef] = {}
    for key in g: z3_constraints[key] = [_sympy_poly_to_z3(var_z3_dict, p).r >= 0 for p in g[key]]
    # Alternative to find values of g when solving
    # for key in g: z3_constraints[key] = [z3.And(Complex('g' + key + str(idx)) == _sympy_poly_to_z3(var_z3_dict, poly), Complex('g' + key + str(idx)).r >= 0) for idx, poly in enumerate(g[key])]
    run_checks(z3_barriers, z3_diffs, z3_changes, z3_k_diffs, z3_constraints, k=k, eps=eps, gamma=gamma, timeout=timeout)


def run_checks(z3_barriers, z3_diffs, z3_changes, z3_k_diffs, z3_constraints, k=1, eps=0.01, gamma=0.01, timeout=300):
    d = calculate_d(k, eps, gamma)
    constraint_val = Complex('constraint_val')

    logger.info("Check " + INIT)
    run_solver([z3.And(z3_constraints[INIT]), constraint_val == z3_barriers[0][1], constraint_val.r > 0], tool=DREAL, timeout=timeout)

    for unitary, z3_barrier in z3_barriers:
        logger.info("Check " + UNSAFE + " for unitary\n" + str(unitary))
        run_solver([z3.And(z3_constraints[UNSAFE]), constraint_val == z3_barrier, constraint_val.r < d], tool=DREAL, timeout=timeout)

    for idx, z3_diff in enumerate(z3_diffs):
        logger.info("Check " + DIFF + str(idx))
        run_solver([z3.And(z3_constraints[INVARIANT]), constraint_val == z3_diff, constraint_val.r > eps], tool=DREAL, timeout=timeout)
    
    for idx, z3_change in enumerate(z3_changes):
        logger.info("Check " + CHANGE + str(idx))
        run_solver([z3.And(z3_constraints[INVARIANT]), constraint_val == z3_change, constraint_val.r > gamma], tool=DREAL, timeout=timeout)

    for idx, z3_k_diff in enumerate(z3_k_diffs):
        logger.info("Check " + INDUCTIVE + str(idx))
        run_solver([z3.And(z3_constraints[INVARIANT]), constraint_val == z3_k_diff, constraint_val.r > 0], tool=DREAL, timeout=timeout)
    logger.info("All constraints checked.")


def run_solver(conds : list[z3.ExprRef], tool=Z3, delta=0.001, timeout=300):
    tactic = z3.Then('solve-eqs','smt')
    s = tactic.solver()
    s.set(timeout=timeout)
    [s.add(z3.simplify(cond)) for cond in conds]
    logger.debug("Conditions in solver:\n" + str(s))
    sat = ""

    if tool == Z3:
        logger.info("Using Z3")
        sat = s.check()
        model = str(s.model()) if sat == z3.sat else str([])
    elif tool == DREAL:
        logger.info("Using dreal")
        sat, model = run_dreal(s, delta=delta, log_level=logger.level, timeout=timeout)
    else: raise_error("No valid tool selected. Use Z3 or dReal")

    if sat in [z3.unsat, DREAL_UNSAT]: logger.info("Constraint satisfied.")
    elif sat in [z3.sat, DREAL_SAT]: raise_error("Counter example: " + model)
    elif sat in [z3.unknown, DREAL_UNKOWN]: logger.warning("Solver returned unkown. Function may not satisfy barrier certificate constraint.")


# Based on: https://stackoverflow.com/a/38980538/19768075
def _sympy_poly_to_z3(var_map, e) -> z3.ExprRef:
    rv = None
    if isinstance(e, sym.Poly): return _sympy_poly_to_z3(var_map, e.as_expr())
    elif not isinstance(e, sym.Expr): raise_error("Expected sympy Expr: " + repr(e))
    if isinstance(e, sym.Symbol): rv = var_map[e]
    elif isinstance(e, sym.Number): rv = float(e)
    elif isinstance(e, sym.Mul): rv = reduce((lambda x, y: x * y), [_sympy_poly_to_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Add): rv = sum([_sympy_poly_to_z3(var_map, exp) for exp in e.args])
    elif isinstance(e, sym.Pow): rv = _sympy_poly_to_z3(var_map, e.args[0]) ** int(_sympy_poly_to_z3(var_map, e.args[1]))
    elif isinstance(e, sym.conjugate): rv = _sympy_poly_to_z3(var_map, e.args[0]).conj()
    elif isinstance(e, sym.core.numbers.ImaginaryUnit): rv = I
    if rv is None: raise_error("Unable to handle input " + repr(e) + " (" + str(type(e)) + ")")
    return rv