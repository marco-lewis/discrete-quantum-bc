from log_settings import LoggerWriter
from utils import *

import logging
import sys

from iteration_utilities import grouper

logger = logging.getLogger("direct")
picos_logger = logging.getLogger("picos")

def direct_method(circuit : List[np.ndarray],
                  g : Dict[str, List[sym.Poly]],
                  Z : List[sym.Symbol],
                  barrier_degree=2,
                  eps=0.01,
                  k=1,
                  verbose=0,
                  log_level=logging.INFO):
    logger.setLevel(log_level)
    picos_logger.setLevel(log_level)
    
    variables = Z + [z.conjugate() for z in Z]
    d = calculate_d(k, eps)
    unitaries = np.unique(circuit, axis=0)
    logger.info("Barrier degree: " + str(barrier_degree) + ", eps: " + str(eps) + ", k: " + str(k) + ", d: " + str(d))

    # 1. Make polynomials
    lams : Dict(str, sym.Poly) = {}
    sym_polys = {}
    sym_poly_eq = dict([
        (INIT,lambda B, lams, g: sym.poly(-B - np.dot(lams, g[INIT]), variables)),
        (UNSAFE,lambda B, lams, g: sym.poly(B - d - np.dot(lams, g[UNSAFE]), variables)),
        (DIFF,lambda B, f, lams, g: sym.poly(-B.subs(zip(Z, np.dot(f, Z))) + B - np.dot(lams, g[INVARIANT]) + eps, variables)),
        (INDUCTIVE,lambda B, fk, lams, g: sym.poly(-B.subs(zip(Z, np.dot(fk, Z)))) + B - np.dot(lams, g[INVARIANT])),
        ])
    
    barrier = create_polynomial(variables, deg=barrier_degree, coeff_tok='b')
    logger.info("Barrier made.")
    logger.debug(barrier)

    logger.info("Making HSOS polynomials...")
    # 1a. Initial condition
    lams[INIT] = [[create_polynomial(variables, deg=g[INIT][i].total_degree(), coeff_tok='s_' + INIT + str(i)+';') for i in range(len(g[INIT]))]]
    sym_polys[INIT] = [sym_poly_eq[INIT](barrier, lams[INIT][0], g)]
    logger.info("Polynomial for " + INIT + " made.")

    # 1b. Unsafe condition
    lams[UNSAFE] = [[create_polynomial(variables, deg=g[UNSAFE][i].total_degree(), coeff_tok='s_' + UNSAFE + str(i)+';') for i in range(len(g[UNSAFE]))]]
    sym_polys[UNSAFE] = [sym_poly_eq[UNSAFE](barrier, lams[UNSAFE][0], g)]
    logger.info("Polynomial for " + UNSAFE + " made.")

    # 1c. Diff conditions
    lams[INVARIANT] = []
    sym_polys[DIFF] = []
    u = 0
    for unitary in unitaries:
        lam = [create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + INVARIANT + str(u) +';' + str(i)) for i in range(len(g[INVARIANT]))]
        lams[INVARIANT].append(lam)
        sym_polys[DIFF].append(sym_poly_eq[DIFF](barrier, unitary, lam, g))
        u += 1
    logger.info("Polynomials for " + DIFF + " made.")

    # 1d. Inductive conditions
    lams[INDUCTIVE] = []
    sym_polys[INDUCTIVE] = []
    circuit_chunks = []
    for circuit_chunk in grouper(circuit, k):
        unitary_k = circuit_chunk[0]
        for unitary in circuit_chunk[1:]: unitary_k = np.dot(unitary, unitary_k)
        circuit_chunks.append(unitary_k)
    u = 0
    for circuit_chunk in circuit_chunks:
        lam = [create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + INDUCTIVE + str(u) +';' + str(i)) for i in range(len(g[INVARIANT]))]
        lams[INDUCTIVE].append(lam)
        sym_polys[INDUCTIVE].append(sym_poly_eq[INDUCTIVE](barrier, unitary_k, lam, g))
        u += 1
    logger.info("Polynomials for " + INDUCTIVE + " made.")
    logger.info("HSOS polynomials made.")
    logger.debug(sym_polys)

    # 2. Get coefficients out to make symbol dictionary
    logger.info("Fetching coefficients.")
    lam_coeffs : Dict[str, List(sym.Symbol)] = {}
    for key in lams: 
        lam_coeffs[key] = []
        for lam in lams[key]:
            lam_coeffs[key] += flatten([[next(iter(coeff.free_symbols)) for coeff in l.coeffs()] for l in lam])

    barrier_coeffs = [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]

    symbol_var_dict : Dict[sym.Symbol, picos.ComplexVariable]= {}
    for lam_symbols in lam_coeffs.values(): symbol_var_dict.update(symbols_to_cvx_var_dict(lam_symbols))
    symbol_var_dict.update(symbols_to_cvx_var_dict(barrier_coeffs))
    logger.info("Symbol to variable dictionary made.")
    logger.debug(symbol_var_dict)

    # 3. Get matrix polynomial and constraints for semidefinite format
    cvx_constraints = []
    cvx_matrices : List[picos.HermitianVariable] = []

    logger.info("Generating lam constraints...")
    for key in lams:
        u = 0
        for lam in lams[key]:
            i = 0
            for poly in lam:
                S_CVX, lam_constraints = PSD_constraint_generator(poly, symbol_var_dict, matrix_name='LAM_' + str(key) + str(u) + ';' + str(i), variables=variables)
                cvx_matrices.append(S_CVX)
                cvx_constraints += lam_constraints
                logger.info(str(key) + str(u) + ';' + str(i) + " done.")
                i += 1
            u += 1
    logger.info("lam constraints generated.")

    logger.info("Generating polynomial constraints...")
    for key in sym_polys:
        u = 0
        for sym_poly in sym_polys[key]:
            Q_CVX, poly_constraint = PSD_constraint_generator(sym_poly, symbol_var_dict, matrix_name='POLY_' + str(key) + str(u), variables=variables)
            cvx_matrices.append(Q_CVX)
            cvx_constraints += poly_constraint
            u += 1
        logger.info(str(key) + " done.")
    logger.info("Poly constraints generated.")

    logger.info("Generating semidefinite constraints...")
    cvx_constraints += [M >> 0 for M in cvx_matrices]
    logger.info("Semidefinite constraints generated.")
    logger.info("Constraints generated")
    logger.info("Number of matrices: " + str(len(cvx_matrices)))
    for M in cvx_matrices: logger.debug(M.name + ": " + str(M.shape))
    logger.info("Number of constraitns: " + str(len(cvx_constraints)))
    logger.debug(cvx_constraints)

    # 4. Solve using PICOS
    prob = picos.Problem()
    prob.minimize = picos.Constant(0)
    for constraint in cvx_constraints: prob.add_constraint(constraint)

    logger.info("Solving problem...")
    sys.stdout = LoggerWriter(picos_logger.info)
    sys.stderr = LoggerWriter(picos_logger.error)
    prob.solve(verbose=bool(verbose))
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    logger.info("Problem status: " + prob.status)
    if "infeasible" in prob.status or "unbounded" in prob.status:
        logging.error("Cannot get barrier from problem.")
        return sym.core.numbers.Infinity
    logger.info("Solution found.")

    # 5. Return the barrier in a readable format
    logger.info("Fetching values...")
    symbols : List[sym.Symbol] = barrier_coeffs + lam_coeffs[INIT] + lam_coeffs[UNSAFE] + lam_coeffs[INVARIANT] + lam_coeffs[INDUCTIVE]
    symbols = list(set(symbols))
    symbols.sort(key = lambda symbol: symbol.name)
    symbol_values = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
    for key in symbol_values:
        if not(symbol_values[key]): t = 0 
        else:
            try:
                t = symbol_values[key].real if abs(symbol_values[key].real) > 1e-10 else 0
                t += symbol_values[key].imag if abs(symbol_values[key].imag) > 1e-10 else 0
            except:
                t = symbol_values[key] if abs(symbol_values[key]) > 1e-10 else 0
        symbol_values[key] = t
    for key in lams:
        for ls in lams[key]:
            for l in ls:
                logger.debug(l.subs(symbol_values))
    for m in cvx_matrices:
        logger.debug(m.name)
        logger.debug(m)
    barrier = barrier.subs(symbol_values)
    logger.info("Barrier made.")
    return barrier