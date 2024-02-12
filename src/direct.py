from src.check import check_barrier
from src.log_settings import LoggerWriter
from src.utils import *

import logging
import sys

from iteration_utilities import grouper

logger = logging.getLogger("direct")
picos_logger = logging.getLogger("picos")

# TODO: Make loops cleaner using enumerate, unpacking, ...
def direct_method(circuit : list[np.ndarray],
                  g : dict[str, list[sym.Poly]],
                  Z : list[sym.Symbol],
                  barrier_degree=2,
                  eps=0.01,
                  gamma=0.01,
                  k=1,
                  verbose=0,
                  log_level=logging.INFO,
                  precision_bound=1e-10,
                  solver='cvxopt',
                  check=False,):
    logger.setLevel(log_level)
    picos_logger.setLevel(log_level)
    
    d = calculate_d(k, eps, gamma)
    variables = generate_variables(Z)
    unitaries : list[np.ndarray] = list(np.unique(circuit, axis=0))
    unitary_idxs = get_unitary_idxs(circuit, unitaries)

    logger.info("Barrier degree: " + str(barrier_degree) +
                ", k: " + str(k) +
                ", eps: " + str(eps) +
                ", gamma: " + str(gamma) +
                ", d: " + str(d))

    # 1. Make polynomials
    barriers : list[sym.Poly] = [create_polynomial(variables, deg=barrier_degree, coeff_tok='b' + str(j) + '_') for j in range(len(unitaries))]
    logger.info("Barriers made.")
    logger.debug(barriers)
    
    idx_pairs = make_idx_pairs(circuit, unitary_idxs)
    chunks : list[tuple[np.ndarray,int,int]] = make_chunks(circuit, unitaries, k)

    lams : dict[str, list[list[sym.Poly]]] = {}
    sym_polys : dict[str, list[sym.Poly]] = {}
    sym_poly_eq = dict([
        (INIT, lambda B, lam, g: sym.poly(-B - np.dot(lam, g[INIT]), variables)),
        (UNSAFE, lambda B, lam, g: sym.poly(B - d - np.dot(lam, g[UNSAFE]), variables)),
        (DIFF, lambda B, f, lam, g: sym.poly(-B.subs(zip(Z, np.dot(f, Z))) + B - np.dot(lam, g[INVARIANT]) + eps, variables)),
        (CHANGE, lambda B, Bnext, lam, g: sym.poly(-Bnext + B - np.dot(lam, g[INVARIANT]) + gamma, variables)),
        (INDUCTIVE, lambda B, Bk, fk, lam, g: sym.poly(-Bk.subs(zip(Z, np.dot(fk, Z))) + B - np.dot(lam, g[INVARIANT]), variables)),
        ])
    
    logger.info("Making HSOS polynomials...")
    # 1a. Initial condition
    lams[INIT] = [[create_polynomial(variables, deg=g[INIT][i].total_degree(), coeff_tok='s_' + INIT + ';' + str(i) + 'c') for i in range(len(g[INIT]))]]
    sym_polys[INIT] = [sym_poly_eq[INIT](barriers[0], lams[INIT][0], g)]
    logger.info("Polynomial for " + INIT + " made.")
    logger.debug(sym_polys[INIT])

    # 1b. Unsafe conditions
    lams[UNSAFE] = [[create_polynomial(variables, deg=g[UNSAFE][i].total_degree(), coeff_tok='s_' + UNSAFE + str(j) + ';' + str(i) + 'c') for i in range(len(g[UNSAFE]))] for j in range(len(unitaries))]
    sym_polys[UNSAFE] = [sym_poly_eq[UNSAFE](barriers[j], lams[UNSAFE][j], g) for j in range(len(unitaries))]
    logger.info("Polynomial for " + UNSAFE + " made.")
    logger.debug(sym_polys[UNSAFE])

    # 1c. Diff conditions
    lams[INVARIANT] = [[create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + INVARIANT + str(j) +';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for j in range(len(unitaries))]
    sym_polys[DIFF] = [sym_poly_eq[DIFF](barriers[j], unitaries[j], lams[INVARIANT][j], g) for j in range(len(unitaries))]
    logger.info("Polynomials for " + DIFF + " made.")
    logger.debug(sym_polys[DIFF])

    # 1d. Change conditions
    lams[CHANGE] = []
    sym_polys[CHANGE] = []
    for (idx, next_idx) in idx_pairs:
        lam = [create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + CHANGE + str(idx) + "," + str(next_idx) + ';' + str(i) + 'c') for i in range(len(g[INVARIANT]))]
        lams[CHANGE].append(lam)
        sym_polys[CHANGE].append(sym_poly_eq[CHANGE](barriers[idx], barriers[next_idx], lam, g))
    logger.info("Polynomials for " + CHANGE + " made.")
    logger.debug(sym_polys[CHANGE])

    # 1e. Inductive conditions
    lams[INDUCTIVE] = []
    sym_polys[INDUCTIVE] = []
    for chunk_id, (unitary_k, fst_idx, last_idx) in enumerate(chunks):
        lam = [create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + INDUCTIVE + str(chunk_id) + ';' + str(i) + 'c') for i in range(len(g[INVARIANT]))]
        lams[INDUCTIVE].append(lam)
        sym_polys[INDUCTIVE].append(sym_poly_eq[INDUCTIVE](barriers[fst_idx], barriers[last_idx], unitary_k, lam, g))
    logger.info("Polynomials for " + INDUCTIVE + " made.")
    logger.debug(sym_polys[INDUCTIVE])
    logger.info("HSOS polynomials made.")

    # 2. Get coefficients out to make symbol dictionary
    # TODO: Fix typing in this section (optional)
    logger.info("Fetching coefficients.")
    lam_coeffs = get_lam_coeffs(lams)
    barrier_coeffs = get_barrier_coeffs(barriers)
    symbol_var_dict = make_symbol_var_dict(lam_coeffs, barrier_coeffs)
    logger.info("Symbol to variable dictionary made.")
    logger.debug(symbol_var_dict)

    # 3. Get matrix polynomial and constraints for semidefinite format
    cvx_constraints = []
    cvx_matrices : list[picos.HermitianVariable] = []

    logger.info("Generating lam constraints...")
    for key in lams:
        for lam_idx, lam in enumerate(lams[key]):
            for lam_poly_idx, poly in enumerate(lam):
                S_CVX, lam_constraints = PSD_constraint_generator(poly, symbol_var_dict, matrix_name='LAM_' + str(key) + str(lam_idx) + ';' + str(lam_poly_idx), variables=variables)
                cvx_matrices.append(S_CVX)
                cvx_constraints += lam_constraints
                logger.info(str(key) + str(lam_idx) + ';' + str(lam_poly_idx) + " done.")
    logger.info("lam constraints generated.")

    logger.info("Generating polynomial constraints...")
    for key in sym_polys:
        for poly_idx, sym_poly in enumerate(sym_polys[key]):
            Q_CVX, poly_constraint = PSD_constraint_generator(sym_poly, symbol_var_dict, matrix_name='POLY_' + str(key) + str(poly_idx), variables=variables)
            cvx_matrices.append(Q_CVX)
            cvx_constraints += poly_constraint
        logger.info(str(key) + " done.")
    logger.info("Polynomial constraints generated.")

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
    # TODO: Add type/infty/symPoly
    fail_barriers = [(unitary, 0) for unitary in unitaries]
    try:
        sys.stdout = LoggerWriter(picos_logger.info)
        sys.stderr = LoggerWriter(picos_logger.error)
        prob.solve(verbose=bool(verbose), solver=solver)
    except Exception as e:
        logger.exception(e)
        return fail_barriers
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    logger.info("Problem status: " + prob.status)
    if "infeasible" in prob.status or "unbounded" in prob.status:
        logger.error("Cannot get barrier from problem.")
        return fail_barriers
    logger.info("Solution found.")

    # 5. Return the barrier in a readable format
    logger.info("Fetching values...")
    symbols : list[sym.Symbol] = barrier_coeffs + lam_coeffs[INIT] + lam_coeffs[UNSAFE] + lam_coeffs[INVARIANT] + lam_coeffs[INDUCTIVE]
    symbols = list(set(symbols))
    symbols.sort(key = lambda symbol: symbol.name)
    symbol_values = get_symbol_values(symbols, symbol_var_dict, precision_bound)

    debug_print_lambda(lams, symbol_values)
    debug_print_matrices(cvx_matrices)

    unitary_barrier_pairs = make_unitary_barrier_pairs(barriers, symbol_values, unitaries)
    logger.info("Barriers made.")
    [logger.info(str(u) + ":\n" + str(b)) for u, b in unitary_barrier_pairs]
    
    # 6. Check barrier works
    if check:
        logger.info("Performing checks")
        check_barrier(unitary_barrier_pairs, g, Z, idx_pairs, chunks, k, eps, gamma, log_level=log_level)
    return unitary_barrier_pairs

def get_unitary_idxs(circuit : list[np.ndarray], unitaries : list[np.ndarray]):
    unitary_idxs = []
    for c in circuit:
        for ui in range(len(unitaries)):
            if np.array_equal(unitaries[ui], c):
                unitary_idxs.append(ui) 
                break
    return unitary_idxs

def make_idx_pairs(circuit: list[np.ndarray], unitary_idxs : list[int]):
    idx_pairs = []
    for i in range(len(circuit)-1):
        idx = unitary_idxs[i]
        next_idx = unitary_idxs[i+1]
        if (idx, next_idx) not in idx_pairs and idx != next_idx: idx_pairs.append((idx, next_idx))
    return idx_pairs

def make_chunks(circuit: list[np.ndarray], unitaries: list[np.ndarray], k):
    circuit_divided : list[tuple[np.ndarray]] = list(grouper(circuit, k))
    unique_chunks : list[tuple[np.ndarray]] = [circuit_divided[0]]
    for circuit_chunk in circuit_divided:
        if circuit_chunk not in unique_chunks: unique_chunks.append(circuit_chunk)

    chunks = []
    for circuit_chunk in unique_chunks:
        unitary_k = circuit_chunk[0]
        for unitary in circuit_chunk[1:]: unitary_k = np.dot(unitary, unitary_k)
        us = [u.tolist() for u in unitaries]
        chunk = (unitary_k, us.index(circuit_chunk[0].tolist()), us.index(circuit_chunk[-1].tolist()))
        chunks.append(chunk)
    return chunks

def get_lam_coeffs(lams : dict[str, list[list[sym.Poly]]]) -> dict[str, list[sym.Symbol]]:
    lam_coeffs = {}
    for key in lams: 
        lam_coeffs[key] = []
        for lam in lams[key]: lam_coeffs[key] += flatten([[next(iter(coeff.free_symbols)) for coeff in l.coeffs()] for l in lam])
    return lam_coeffs

def get_barrier_coeffs(barriers : list[sym.Poly]) -> list[sym.Symbol]:
    barrier_coeffs = []
    for barrier in barriers: barrier_coeffs += [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]
    return barrier_coeffs

def symbols_to_cvx_var_dict(symbols : list[sym.Symbol]):
    cvx_vars = [picos.ComplexVariable(name = s.name) for s in symbols]
    symbol_var_dict = dict(zip(symbols, cvx_vars))
    return symbol_var_dict

def make_symbol_var_dict(lam_coeffs, barrier_coeffs) -> dict[sym.Symbol, picos.ComplexVariable]:
    symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable]= {}
    for lam_symbols in lam_coeffs.values(): symbol_var_dict.update(symbols_to_cvx_var_dict(lam_symbols))
    symbol_var_dict.update(symbols_to_cvx_var_dict(barrier_coeffs))
    return symbol_var_dict

def get_symbol_values(symbols : list[sym.Symbol], symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable], precision_bound : float) -> dict[sym.Symbol, complex]:
    symbol_values : dict[sym.Symbol, complex] = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
    for key in symbol_values:
        if not(symbol_values[key]): t = 0
        else:
            try:
                t = symbol_values[key].real if abs(symbol_values[key].real) > precision_bound else 0
                t += 1j*symbol_values[key].imag if abs(symbol_values[key].imag) > precision_bound else 0
            except:
                t = symbol_values[key] if abs(symbol_values[key]) > precision_bound else 0
        symbol_values[key] = t
    return symbol_values

def debug_print_lambda(lams, symbol_values):
    logger.debug("lambda polynomials")
    for key in lams:
        for idx, ls in enumerate(lams[key]):
            for jdx, poly in enumerate(ls):
                logger.debug(key + str(idx) + ";" + str(jdx))
                logger.debug(poly.subs(symbol_values))

def debug_print_matrices(cvx_matrices):
    logger.debug("Convex Matrices")
    for m in cvx_matrices:
        logger.debug(m.name)
        logger.debug(m)

def make_unitary_barrier_pairs(barriers, symbol_values, unitaries) -> list[tuple[np.ndarray, sym.Poly]]:
    barriers = [barrier.subs(symbol_values) for barrier in barriers]
    return list(zip(unitaries, barriers))