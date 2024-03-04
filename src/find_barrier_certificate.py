from src.check import check_barrier
from src.log_settings import LoggerWriter
from src.typings import *
from src.utils import *

from collections import defaultdict
import time
import logging
import sys

from iteration_utilities import grouper

logger = logging.getLogger("findbc")
picos_logger = logging.getLogger("picos")

# TODO: Make loops cleaner using enumerate, unpacking, ...
def find_barrier_certificate(circuit : Circuit,
                  g : SemiAlgebraicDict,
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
    times : Timings = {}
    times[TIME_SP] = time.time()
    
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
    logger.debug("Index pairs: " + str(idx_pairs))
    logger.debug("Chunks: " + str(chunks))
    sym_poly_eq = dict([
        (INIT, lambda B, lam, g: sym.poly(-B - np.dot(lam, g[INIT]), variables)),
        (UNSAFE, lambda B, lam, g: sym.poly(B - d - np.dot(lam, g[UNSAFE]), variables)),
        (DIFF, lambda B, f, lam, g: sym.poly(-B.subs(zip(Z, np.dot(f, Z))) + B - np.dot(lam, g[INVARIANT]) + eps, variables)),
        (CHANGE, lambda B, Bnext, lam, g: sym.poly(-Bnext + B - np.dot(lam, g[INVARIANT]) + gamma, variables)),
        (INDUCTIVE, lambda B, Bk, fk, lam, g: sym.poly(-Bk.subs(zip(Z, np.dot(fk, Z))) + B - np.dot(lam, g[INVARIANT]), variables)),
        ])
    sym_polys = make_sym_polys(barriers, lams, g, unitaries, idx_pairs, chunks, k, sym_poly_eq)

    # 2. Get coefficients out to make symbol dictionary
    logger.info("Fetching coefficients.")
    lam_coeffs = get_lam_coeffs(lams)
    barrier_coeffs = get_barrier_coeffs(barriers)
    symbol_var_dict = make_symbol_var_dict(lam_coeffs, barrier_coeffs)
    logger.info("Symbol to variable dictionary made.")
    logger.debug(symbol_var_dict)

    # 3. Get matrix polynomial and constraints for semidefinite format
    cvx_matrices, cvx_constraints = get_cvx_format(lams, symbol_var_dict, variables, sym_polys)
    times[TIME_SP] = time.time() - times[TIME_SP]

    # 4. Solve using PICOS
    exit_prog, times[TIME_PICOS] = run_picos(cvx_constraints, solver, verbose)
    if exit_prog: sys.exit(exit_prog)
    
    # 5. Return the barrier in a readable format
    post_time = time.time()
    barrier_certificate = fetch_values(barriers, unitaries, barrier_coeffs, lam_coeffs, symbol_var_dict, precision_bound, lams, cvx_matrices)
    post_time = time.time() - post_time
    times[TIME_SP] += post_time

    # 6. Check barrier works
    times[TIME_VERIF] = 0
    if check:
        logger.info("Performing checks")
        times[TIME_VERIF] = time.time()
        check_barrier(barrier_certificate, g, Z, idx_pairs, chunks, k, eps, gamma, log_level=log_level)
        times[TIME_VERIF] = time.time() - times[TIME_VERIF]
    
    row_msg = lambda m1, m2: f'{m1:<25}{m2}'
    format_time = lambda t: f'{t:.3f}'
    time_message = [
        "Table of runtimes",
        row_msg("Process", "Time (s)"),
        row_msg("Setup + Postprocessing", format_time(times[TIME_SP])),
        row_msg("PICOS", format_time(times[TIME_PICOS])),
        row_msg("Verification", format_time(times[TIME_VERIF])),
    ]
    for msg in time_message: logger.info(msg)

    return barrier_certificate, times

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
        unique = True
        for unique_chunk in unique_chunks:
            unique = unique and not np.all([np.array_equal(m1, m2) for m1, m2 in zip(circuit_chunk, unique_chunk)])
        if unique: unique_chunks.append(circuit_chunk)

    chunks = []
    for circuit_chunk in unique_chunks:
        unitary_k = circuit_chunk[0]
        for unitary in circuit_chunk[1:]: unitary_k = np.dot(unitary, unitary_k)
        us = [u.tolist() for u in unitaries]
        chunk = (unitary_k, us.index(circuit_chunk[0].tolist()), us.index(circuit_chunk[-1].tolist()))
        chunks.append(chunk)
    return chunks

def get_degree(poly : sym.Poly) -> int:
    return poly.total_degree() if poly.total_degree() % 2 == 0 else poly.total_degree() + 1

def make_lambdas(variables : list[sym.Symbol], g : SemiAlgebraicDict, unitaries : Unitaries, idx_pairs : tuple[int, int], chunks : list[Chunk]) -> dict[str, LamList]:
    lams = {}
    lams[INIT] = [[create_polynomial(variables, deg=get_degree(g[INIT][i]), coeff_tok='s_' + INIT + ';' + str(i) + 'c') for i in range(len(g[INIT]))]]
    lams[UNSAFE] = [[create_polynomial(variables, deg=get_degree(g[UNSAFE][i]), coeff_tok='s_' + UNSAFE + str(j) + ';' + str(i) + 'c') for i in range(len(g[UNSAFE]))] for j in range(len(unitaries))]
    lams[DIFF] = [[create_polynomial(variables, deg=get_degree(g[INVARIANT][i]), coeff_tok='s_' + DIFF + str(j) +';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for j in range(len(unitaries))]
    lams[CHANGE] = [[create_polynomial(variables, deg=get_degree(g[INVARIANT][i]), coeff_tok='s_' + CHANGE + str(idx) + "," + str(next_idx) + ';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for (idx, next_idx) in idx_pairs]
    lams[INDUCTIVE] = [[create_polynomial(variables, deg=get_degree(g[INVARIANT][i]), coeff_tok='s_' + INDUCTIVE + str(chunk_id) + ';' + str(i) + 'c') for i in range(len(g[INVARIANT]))] for chunk_id, _ in enumerate(chunks)]
    return lams

def make_sym_polys(barriers : list[Barrier], lams : dict[str, LamList], g : SemiAlgebraicDict, unitaries : Unitaries, idx_pairs : list[tuple[int,int]], chunks : list[Chunk], k : int, sym_poly_eq : dict[callable]) -> dict[str, list[sym.Poly]]:
    sym_polys : dict[str, list[sym.Poly]] = {}    
    logger.info("Making HSOS polynomials...")
    sym_polys[INIT] = [sym_poly_eq[INIT](barriers[0], lams[INIT][0], g)]
    logger.info("Polynomial for " + INIT + " made.")
    logger.debug(sym_polys[INIT])

    sym_polys[UNSAFE] = [sym_poly_eq[UNSAFE](barriers[j], lams[UNSAFE][j], g) for j in range(len(unitaries))]
    logger.info("Polynomials for " + UNSAFE + " made.")
    logger.debug(sym_polys[UNSAFE])

    if k == 1: logger.info("No polynomials for " + DIFF + " need to be made (k=1).")
    else:
        sym_polys[DIFF] = [sym_poly_eq[DIFF](barriers[j], unitaries[j], lams[DIFF][j], g) for j in range(len(unitaries))]
        logger.info("Polynomials for " + DIFF + " made.")
        logger.debug(sym_polys[DIFF])

    if idx_pairs == []: logger.info("No polynomials for " + CHANGE + " need to be made (only one unitary).")
    else:
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

def PSD_constraint_generator(sym_polynomial : sym.Poly,
                             symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable],
                             matrix_name='Q',
                             variables=[]):
    # Setup dictionary of monomials to cvx coefficients for sym_polynomial
    if sym_polynomial.total_degree() % 2 == 1:
        logger.exception("Polynomial does not have degree 2.")
        exit(1)
    cvx_coeffs = convert_exprs(sym_polynomial.coeffs(), symbol_var_dict)
    poly_monom_to_cvx = dict(zip(sym_polynomial.monoms(), cvx_coeffs))
    poly_monom_to_cvx = defaultdict(lambda: 0.0, poly_monom_to_cvx)

    # Create sympy matrix and quadratic form as polynomial
    m = create_polynomial(variables[:len(variables)//2], deg=sym_polynomial.total_degree()//2, monomial=True)
    vector_monomials = np.array([np.prod([x**k for x, k in zip(m.gens, mon)]) for mon in m.monoms()])
    num_of_monom = len(vector_monomials)
    Q_SYM = sym.MatrixSymbol(matrix_name, num_of_monom, num_of_monom)
    Q_QUAD = sym.poly(vector_monomials.conj().T @ Q_SYM @ vector_monomials, variables)

    # Create cvx matrix and dictionary of monomials to cvx matrix terms
    Q_CVX = picos.HermitianVariable(name=matrix_name, shape=(num_of_monom, num_of_monom))
    Q_cvx_coeffs = convert_exprs_of_matrix(Q_QUAD.coeffs(), Q_CVX)
    Q_monom_to_cvx = dict(zip(Q_QUAD.monoms(), Q_cvx_coeffs))

    # Link matrix variables to polynomial variables
    constraints = [Q_monom_to_cvx[key] == poly_monom_to_cvx[key] for key in Q_monom_to_cvx]
    # Next line needed if using //2 for degree in m to capture remaining terms
    constraints += [poly_monom_to_cvx[key] == 0 for key in list(filter(lambda k: k not in Q_monom_to_cvx.keys(), poly_monom_to_cvx.keys()))]
    return Q_CVX, constraints

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

def run_picos(cvx_constraints : list[picos.constraints.Constraint], solver : str, verbose : int) -> tuple[int, float]:
    prob = picos.Problem()
    prob.minimize = picos.Constant(0)
    for constraint in cvx_constraints: prob.add_constraint(constraint)

    logger.info("Solving problem...")
    try:
        sys.stdout = LoggerWriter(picos_logger.info)
        sys.stderr = LoggerWriter(picos_logger.error)
        picos_time = time.time()
        prob.solve(verbose=bool(verbose), solver=solver)
        picos_time = time.time() - picos_time
    except Exception as e:
        logger.exception(e)
        return 1, 0
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    logger.info("Problem status: " + prob.status)
    if "infeasible" in prob.status or "unbounded" in prob.status:
        logger.error("Cannot get barrier from problem.")
        return 1, 0
    logger.info("Solution found.")
    return 0, picos_time

def get_symbol_values(symbols : list[sym.Symbol], symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable], precision_bound : float) -> dict[sym.Symbol, complex]:
    symbol_values : dict[sym.Symbol, complex] = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
    ndigits = -int(np.log10(precision_bound))
    for key in symbol_values:
        if not(symbol_values[key]): t = 0
        else:
            try:
                t = round(symbol_values[key].real, ndigits) if abs(symbol_values[key].real) > precision_bound else 0
                t += 1j*round(symbol_values[key].imag, ndigits) if abs(symbol_values[key].imag) > precision_bound else 0
            except:
                t = round(symbol_values[key], ndigits) if abs(symbol_values[key]) > precision_bound else 0
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