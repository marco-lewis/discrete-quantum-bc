from src.check import check_barrier
from src.log_settings import LoggerWriter
from src.typings import *
from src.utils import *

import logging
import sys

from iteration_utilities import grouper

logger = logging.getLogger("direct")
picos_logger = logging.getLogger("picos")

# TODO: Make loops cleaner using enumerate, unpacking, ...
def find_barrier_certificate(circuit : Circuit,
                  g : SemiAlgebraic,
                  Z : list[sym.Symbol],
                  barrier_degree=2,
                  eps=0.01,
                  gamma=0.01,
                  k=1,
                  verbose=0,
                  log_level=logging.INFO,
                  precision_bound=1e-10,
                  solver='cvxopt',
                  check=False) -> BarrierCertificate:
    logger.setLevel(log_level)
    picos_logger.setLevel(log_level)
    
    d = calculate_d(k, eps, gamma)
    variables = generate_variables(Z)
    unitaries : Unitaries = list(np.unique(circuit, axis=0))
    unitary_idxs = get_unitary_idxs(circuit, unitaries)

    logger.info("Barrier degree: " + str(barrier_degree) +
                ", k: " + str(k) +
                ", eps: " + str(eps) +
                ", gamma: " + str(gamma) +
                ", d: " + str(d))

    # 1. Make polynomials
    barriers : list[Barrier] = [create_polynomial(variables, deg=barrier_degree, coeff_tok='b' + str(j) + '_') for j in range(len(unitaries))]
    logger.info("Barriers made.")
    logger.debug(barriers)
    
    logger.info("Setting up lambda terms and utilities...")
    idx_pairs = make_idx_pairs(circuit, unitary_idxs)
    chunks = make_chunks(circuit, unitaries, k)
    lams = make_lambdas(variables, g, unitaries, idx_pairs, chunks)
    logger.info("Lambda functions and utilities set up.")
    sym_poly_eq = dict([
        (INIT, lambda B, lam, g: sym.poly(-B - np.dot(lam, g[INIT]), variables)),
        (UNSAFE, lambda B, lam, g: sym.poly(B - d - np.dot(lam, g[UNSAFE]), variables)),
        (DIFF, lambda B, f, lam, g: sym.poly(-B.subs(zip(Z, np.dot(f, Z))) + B - np.dot(lam, g[INVARIANT]) + eps, variables)),
        (CHANGE, lambda B, Bnext, lam, g: sym.poly(-Bnext + B - np.dot(lam, g[INVARIANT]) + gamma, variables)),
        (INDUCTIVE, lambda B, Bk, fk, lam, g: sym.poly(-Bk.subs(zip(Z, np.dot(fk, Z))) + B - np.dot(lam, g[INVARIANT]), variables)),
        ])
    sym_polys = make_sym_polys(barriers, lams, g, unitaries, idx_pairs, chunks, sym_poly_eq)

    # 2. Get coefficients out to make symbol dictionary
    logger.info("Fetching coefficients.")
    lam_coeffs = get_lam_coeffs(lams)
    barrier_coeffs = get_barrier_coeffs(barriers)
    symbol_var_dict = make_symbol_var_dict(lam_coeffs, barrier_coeffs)
    logger.info("Symbol to variable dictionary made.")
    logger.debug(symbol_var_dict)

    # 3. Get matrix polynomial and constraints for semidefinite format
    cvx_matrices, cvx_constraints = get_cvx_format(lams, symbol_var_dict, variables, sym_polys)

    # 4. Solve using PICOS
    exit_prog = run_picos(cvx_constraints, solver, verbose)
    if exit_prog: sys.exit(exit_prog)
    
    # 5. Return the barrier in a readable format
    barrier_certificate = fetch_values(barriers, unitaries, barrier_coeffs, lam_coeffs, symbol_var_dict, precision_bound, lams, cvx_matrices)

    # 6. Check barrier works
    if check:
        logger.info("Performing checks")
        check_barrier(barrier_certificate, g, Z, idx_pairs, chunks, k, eps, gamma, log_level=log_level)
    return barrier_certificate

def get_unitary_idxs(circuit : Circuit, unitaries : Unitaries) -> list[Idx]:
    unitary_idxs = []
    for c in circuit:
        for ui in range(len(unitaries)):
            if np.array_equal(unitaries[ui], c):
                unitary_idxs.append(ui) 
                break
    return unitary_idxs

def make_idx_pairs(circuit : Circuit, unitary_idxs : list[Idx]) -> list[tuple[int, int]]:
    idx_pairs = []
    for i in range(len(circuit)-1):
        idx = unitary_idxs[i]
        next_idx = unitary_idxs[i+1]
        if (idx, next_idx) not in idx_pairs and idx != next_idx: idx_pairs.append((idx, next_idx))
    return idx_pairs

def make_chunks(circuit : list[np.ndarray], unitaries : list[np.ndarray], k : int) -> list[Chunk]:
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

def make_lambdas(variables : list[sym.Symbol], g : SemiAlgebraic, unitaries : Unitaries, idx_pairs : tuple[int, int], chunks : list[Chunk]) -> dict[str, LamList]:
    lams = {}
    lams[INIT] = [[create_polynomial(variables, deg=g[INIT][i].total_degree(), coeff_tok='s_' + INIT + ';' + str(i) + 'c') for i in range(len(g[INIT]))]]
    lams[UNSAFE] = [[create_polynomial(variables, deg=g[UNSAFE][i].total_degree(), coeff_tok='s_' + UNSAFE + str(j) + ';' + str(i) + 'c') for i in range(len(g[UNSAFE]))] for j in range(len(unitaries))]
    lams[DIFF] = [[create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + DIFF + str(j) +';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for j in range(len(unitaries))]
    lams[CHANGE] = [[create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + CHANGE + str(idx) + "," + str(next_idx) + ';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for (idx, next_idx) in idx_pairs]
    lams[INDUCTIVE] = [[create_polynomial(variables, deg=g[INVARIANT][i].total_degree(), coeff_tok='s_' + INDUCTIVE + str(chunk_id) + ';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for chunk_id, _ in enumerate(chunks)]
    return lams

def make_sym_polys(barriers : list[Barrier], lams : dict[str, LamList], g : SemiAlgebraic, unitaries : Unitaries, idx_pairs : list[tuple[int,int]], chunks : list[Chunk], sym_poly_eq) -> dict[str, list[sym.Poly]]:
    sym_polys : dict[str, list[sym.Poly]] = {}    
    logger.info("Making HSOS polynomials...")
    sym_polys[INIT] = [sym_poly_eq[INIT](barriers[0], lams[INIT][0], g)]
    logger.info("Polynomial for " + INIT + " made.")
    logger.debug(sym_polys[INIT])

    sym_polys[UNSAFE] = [sym_poly_eq[UNSAFE](barriers[j], lams[UNSAFE][j], g) for j in range(len(unitaries))]
    logger.info("Polynomial for " + UNSAFE + " made.")
    logger.debug(sym_polys[UNSAFE])

    sym_polys[DIFF] = [sym_poly_eq[DIFF](barriers[j], unitaries[j], lams[DIFF][j], g) for j in range(len(unitaries))]
    logger.info("Polynomials for " + DIFF + " made.")
    logger.debug(sym_polys[DIFF])

    sym_polys[CHANGE] = [sym_poly_eq[CHANGE](barriers[idx], barriers[next_idx], lam, g) for (idx, next_idx), lam in zip(idx_pairs, lams[CHANGE])]
    logger.info("Polynomials for " + CHANGE + " made.")
    logger.debug(sym_polys[CHANGE])

    sym_polys[INDUCTIVE] = [sym_poly_eq[INDUCTIVE](barriers[fst_idx], barriers[last_idx], unitary_k, lam, g) for (unitary_k, fst_idx, last_idx), lam in zip(chunks, lams[INDUCTIVE])]
    logger.info("Polynomials for " + INDUCTIVE + " made.")
    logger.debug(sym_polys[INDUCTIVE])
    logger.info("HSOS polynomials made.")
    return sym_polys

def get_lam_coeffs(lams : dict[str, LamList]) -> dict[str, list[sym.Symbol]]:
    lam_coeffs = {}
    for key in lams: 
        lam_coeffs[key] = []
        for lam in lams[key]: lam_coeffs[key] += flatten([[next(iter(coeff.free_symbols)) for coeff in l.coeffs()] for l in lam])
    return lam_coeffs

def get_barrier_coeffs(barriers : list[Barrier]) -> list[sym.Symbol]:
    barrier_coeffs = []
    for barrier in barriers: barrier_coeffs += [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]
    return barrier_coeffs

def symbols_to_cvx_var_dict(symbols : list[sym.Symbol]) -> dict[sym.Symbol, picos.ComplexVariable]:
    cvx_vars = [picos.ComplexVariable(name = s.name) for s in symbols]
    symbol_var_dict = dict(zip(symbols, cvx_vars))
    return symbol_var_dict

def make_symbol_var_dict(lam_coeffs : dict[str, list[sym.Symbol]], barrier_coeffs : list[sym.Symbol]) -> dict[sym.Symbol, picos.ComplexVariable]:
    symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable]= {}
    for lam_symbols in lam_coeffs.values(): symbol_var_dict.update(symbols_to_cvx_var_dict(lam_symbols))
    symbol_var_dict.update(symbols_to_cvx_var_dict(barrier_coeffs))
    return symbol_var_dict

def get_cvx_format(lams : dict[str, LamList], symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable], variables : list[sym.Symbol], sym_polys : dict[str, list[sym.Poly]]) -> tuple[list[picos.HermitianVariable], picos.constraints.Constraint]:
    cvx_constraints : list[picos.constraints.Constraint] = []
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
    return cvx_matrices, cvx_constraints

def run_picos(cvx_constraints : list[picos.constraints.Constraint], solver : str, verbose : int) -> int:
    prob = picos.Problem()
    prob.minimize = picos.Constant(0)
    for constraint in cvx_constraints: prob.add_constraint(constraint)

    logger.info("Solving problem...")
    try:
        sys.stdout = LoggerWriter(picos_logger.info)
        sys.stderr = LoggerWriter(picos_logger.error)
        prob.solve(verbose=bool(verbose), solver=solver)
    except Exception as e:
        logger.exception(e)
        return 1
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    logger.info("Problem status: " + prob.status)
    if "infeasible" in prob.status or "unbounded" in prob.status:
        logger.error("Cannot get barrier from problem.")
        return 1
    logger.info("Solution found.")
    return 0

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

def debug_print_lambda(lams : dict[str, LamList], symbol_values : dict[sym.Symbol, complex]):
    logger.debug("lambda polynomials")
    for key in lams:
        for idx, ls in enumerate(lams[key]):
            for jdx, poly in enumerate(ls):
                logger.debug(key + str(idx) + ";" + str(jdx))
                logger.debug(poly.subs(symbol_values))

def debug_print_matrices(cvx_matrices : list[picos.HermitianVariable]):
    logger.debug("Convex Matrices")
    for m in cvx_matrices:
        logger.debug(m.name)
        logger.debug(m)

def get_barrier_certificate_values(barriers, symbol_values, unitaries) -> list[tuple[np.ndarray, sym.Poly]]:
    barriers = [barrier.subs(symbol_values) for barrier in barriers]
    return list(zip(unitaries, barriers))

def fetch_values(barriers : list[Barrier],
                 unitaries : Unitaries,
                 barrier_coeffs : list[sym.Symbol],
                 lam_coeffs : dict[str, list[sym.Symbol]],
                 symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable],
                 precision_bound : float,
                 lams : dict[str, LamList],
                 cvx_matrices : list[picos.HermitianVariable]) -> BarrierCertificate:
    logger.info("Fetching values...")
    symbols : list[sym.Symbol] = barrier_coeffs + lam_coeffs[INIT] + lam_coeffs[UNSAFE] + lam_coeffs[DIFF] + lam_coeffs[INDUCTIVE]
    symbols = list(set(symbols))
    symbols.sort(key = lambda symbol: symbol.name)
    symbol_values = get_symbol_values(symbols, symbol_var_dict, precision_bound)

    debug_print_lambda(lams, symbol_values)
    debug_print_matrices(cvx_matrices)

    barrier_certificate = get_barrier_certificate_values(barriers, symbol_values, unitaries)
    logger.info("Barriers made.")
    [logger.info(str(u) + ":\n" + str(b)) for u, b in barrier_certificate]
    return barrier_certificate