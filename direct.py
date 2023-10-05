from log_settings import LoggerWriter
from utils import *

import logging
import sys

logger = logging.getLogger("direct")
picos_logger = logging.getLogger("picos")

def direct_method(unitary : np.ndarray,
                  g : Dict[str, List[sym.Poly]],
                  Z : List[sym.Symbol],
                  barrier_degree=2,
                  eps=0.01,
                  k=1,
                  verbose=0,
                  log_level=logging.INFO):
    logger.setLevel(log_level)
    
    variables = Z + [z.conjugate() for z in Z]
    d = np.ceil(k * eps) + 1
    logger.info("Barrier degree: " + str(barrier_degree) + ", eps: " + str(eps) + ", k: " + str(k) + ", d: " + str(d))

    # 1. Generate lam, barrier
    lams : Dict(str, sym.Poly) = {}
    for key in g:
        lams[key] = [create_polynomial(variables, deg=g[key][i].total_degree(), coeff_tok='s_' + key + str(i)+';') for i in range(len(g[key]))]
        logger.info("lam polynomial for " + key + " made.")
    lams[INDUCTIVE] = [create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + INDUCTIVE + str(i)+';') for i in range(len(g[INVARIANT]))]
    logger.info("lam polynomial for " + INDUCTIVE + " made.")
    logger.info("lams defined.")
    logger.debug(lams)

    barrier = create_polynomial(variables, deg=barrier_degree, coeff_tok='b')
    logger.info("Barrier made.")
    logger.debug(barrier)

    # 2. Make arbitrary polynomials for SOS terms
    sym_poly_eq = dict([
        (INIT,lambda B, lams, g: sym.poly(-B - np.dot(lams[INIT], g[INIT]), variables)),
        (UNSAFE,lambda B, lams, g: sym.poly(B - d - np.dot(lams[UNSAFE], g[UNSAFE]), variables)),
        (DIFF,lambda B, f, lams, g: sym.poly(-B.subs(zip(Z, np.dot(f, Z))) + B - np.dot(lams[INVARIANT], g[INVARIANT]) + eps, variables)),
        (INDUCTIVE,lambda B, fk, lams, g: sym.poly(-B.subs(zip(Z, np.dot(fk, Z)))) + B - np.dot(lams[INDUCTIVE], g[INVARIANT])),
        ])
    logger.info("Making polynomials.")
    sym_polys = {}
    for key in sym_poly_eq:
        if key == DIFF: sym_polys[key] = sym_poly_eq[key](barrier, unitary, lams, g)
        elif key == INDUCTIVE:
            fk = unitary
            for i in range(1,k): fk = np.dot(unitary, fk)
            sym_polys[key] = sym_poly_eq[key](barrier, fk, lams, g)
        else: sym_polys[key] = sym_poly_eq[key](barrier, lams, g)
        logger.info("Polynomial for " + key + " made.")
    logger.info("Polynomials made.")
    logger.debug(sym_polys)

    lam_coeffs : Dict[str, List(sym.Symbol)] = {}
    for key in lams: lam_coeffs[key] = flatten([[next(iter(coeff.free_symbols)) for coeff in lam.coeffs()] for lam in lams[key]])

    barrier_coeffs = [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]

    symbol_var_dict : Dict[sym.Symbol, picos.ComplexVariable]= {}
    for lam_symbols in lam_coeffs.values(): symbol_var_dict.update(symbols_to_cvx_var_dict(lam_symbols))
    symbol_var_dict.update(symbols_to_cvx_var_dict(barrier_coeffs))

    # 3. Get matrix polynomial and constraints
    cvx_constraints = []
    cvx_matrices : List[picos.HermitianVariable] = []

    logger.info("Generating lam constraints...")
    for key in lams:
        i = 0
        for poly in lams[key]:
            S_CVX, lam_constraints = PSD_constraint_generator(poly, symbol_var_dict, matrix_name='LAM_' + str(key) + str(i), variables=variables)
            cvx_matrices.append(S_CVX)
            cvx_constraints += lam_constraints
            logger.info(str(key) + str(i) + " done.")
            i += 1
    logger.info("lam constraints generated.")

    logger.info("Generating polynomial constraints...")
    for key in sym_polys:
        Q_CVX, poly_constraint = PSD_constraint_generator(sym_polys[key], symbol_var_dict, matrix_name='POLY_' + str(key), variables=variables)
        cvx_matrices.append(Q_CVX)
        cvx_constraints += poly_constraint
        logger.info(str(key) + " done.")
    logger.info("Poly constraints generated.")

    logger.info("Generating semidefinite constraints...")
    cvx_constraints += [M >> 0 for M in cvx_matrices]
    logger.info("Semidefinite constraints generated.")
    logger.info("Constraints generated")
    logger.debug(cvx_constraints)

    # 4. Solve using PICOS
    prob = picos.Problem()
    prob.minimize = picos.Constant(0)
    for constraint in cvx_constraints: prob.add_constraint(constraint)

    picos_logger.info("Solving problem...")
    sys.stdout = LoggerWriter(picos_logger.info)
    sys.stderr = LoggerWriter(picos_logger.error)
    prob.solve(verbose=bool(verbose))
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    picos_logger.info(prob.status)
    if prob.status in ["infeasible", "unbounded"]:
        logging.error("Cannot get barrier from problem.")
        return sym.core.numbers.Infinity

    # 5. Return the barrier in a readable format
    logger.info("Fetching values...")
    symbols : List[sym.Symbol] = barrier_coeffs + lam_coeffs[INIT] + lam_coeffs[UNSAFE] + lam_coeffs[INVARIANT] + lam_coeffs[INDUCTIVE]
    symbols = list(set(symbols))
    symbols.sort(key = lambda symbol: symbol.name)
    symbol_values = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
    for key in symbol_values: symbol_values[key] = symbol_values[key] if abs(symbol_values[key]) > 1e-10 else 0
    for key in lams:
        i = 0
        for lam in lams[key]:
            logger.debug("lam_" + str(key) + str(i), lam.subs(symbol_values))
            i += 1
    for m in cvx_matrices: logger.debug(m, m.value)
    return barrier.subs(symbol_values)