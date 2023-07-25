from collections import defaultdict
import itertools
from typing import Dict, List

import cvxpy as cp
import numpy as np
import sympy as sym
from sympy.matrices.expressions.matexpr import MatrixElement

verbose = 0

UNSAFE = 'unsafe'
INIT = 'init'
INVARIANT = 'invariant'
DIFF = 'diff'
LOC = 'loc'

n = 2
mark = 2
N = 2**n

def flatten(matrix): return [item for row in matrix for item in row]

def to_poly(expr_list, variables, domain=sym.CC):
    return [sym.poly(l, variables, domain=domain) for l in expr_list]

def create_polynomial(variables, deg=2, coeff_tok='a', monomial=False):
    p = []
    for d in range(deg + 1): p += itertools.combinations_with_replacement(variables, d)
    p = [np.prod([t for t in term]) for term in p]
    K = sym.CC
    if monomial: coeffs = [1]*len(p)
    else: 
        coeffs = [sym.Symbol(coeff_tok + str(i), complex=True) for i in range(len(p))]
        K = K[coeffs]
    p = np.sum([coeffs[i] * p[i] for i in range(len(p))])
    return sym.poly(p, variables, domain=K)

def symbols_to_cvx_var(symbols: List[sym.Symbol]):
    cvx_vars = [cp.Variable(name = s.name, complex=True) for s in symbols]
    symbol_var_dict = dict(zip(symbols, cvx_vars))
    return symbol_var_dict

def convert_exprs(exprs: List[sym.Poly], symbol_var_dict:Dict[sym.Symbol, cp.Variable]):
    def convert(expr):
        if isinstance(expr, sym.Add): return np.sum([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Mul): return np.prod([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Pow): return convert(expr.args[0])**convert(expr.args[1])
        if symbol_var_dict.get(expr) is not None: return symbol_var_dict[expr]
        return expr
    return [convert(expr) for expr in exprs]

def convert_exprs_of_matrix(exprs: List[sym.Poly], matrix_to_convert:sym.MatrixSymbol):
    new_matrix = cp.Variable(matrix_to_convert.shape, name=matrix_to_convert.name)
    def convert(expr):
        if isinstance(expr, sym.Add): return np.sum([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Mul): return np.prod([convert(arg) for arg in expr.args])
        if isinstance(expr, MatrixElement): return new_matrix[expr.i, expr.j]
        return expr
    return [convert(expr) for expr in exprs]

def PSD_constraint_generator(sym_polynomial, symbol_var_dict, matrix_name='Q'):
    # Convert sympy polynomial to cvx variables
    cvx_coeffs = convert_exprs(sym_polynomial.coeffs(), symbol_var_dict)
    poly_monom_to_cvx = dict(zip(sym_polynomial.monoms(), cvx_coeffs))
    poly_monom_to_cvx = defaultdict(lambda: 0.0, poly_monom_to_cvx)

    # Create matrix and quadratic form
    m = create_polynomial(variables, deg=int(np.ceil(sym_polynomial.total_degree()/2)), monomial=True)
    vector_monomials = np.array([np.prod([x**k for x, k in zip(m.gens, mon)]) for mon in m.monoms()])
    num_of_monom = len(vector_monomials)
    Q = sym.MatrixSymbol(matrix_name, num_of_monom, num_of_monom)
    vH_Q_v = sym.poly(vector_monomials.conj().T @ Q @ vector_monomials, variables)

    # Create cvx variable matrix
    Q_CVX = cp.Variable((num_of_monom, num_of_monom), hermitian=True, name=matrix_name)
    Q_cvx_coeffs = convert_exprs_of_matrix(vH_Q_v.coeffs(), Q)
    Q_monom_to_cvx = dict(zip(vH_Q_v.monoms(), Q_cvx_coeffs))

    # Link matrix variables to polynomial variables
    constraints = [Q_monom_to_cvx[key] == poly_monom_to_cvx[key] for key in Q_monom_to_cvx]
    return Q_CVX, constraints

# 0. Inputs, variable definitions and constants
eps = 1

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]
dot = lambda lam, g: np.sum([li * gi for li,gi in zip(lam, g)])
poly_eq = dict([
    (INIT,lambda B, lams, g: -B - dot(lams[INIT], g[INIT])),
    (UNSAFE,lambda B, lams, g: B - eps - dot(lams[UNSAFE], g[UNSAFE])),
    (DIFF,lambda dB, lams, g: -dB - dot(lams[INVARIANT], g[INVARIANT])),
    # (LOC,lambda B, lam, g: -B - dot(lam, g)),
    ])


g_u = [
    # -np.prod([0.7 - Z[i] * sym.conjugate(Z[i]) if not(i == mark) else 1 for i in range(N)]),
    Z[mark] * sym.conjugate(Z[mark]) - 0.9,
    1 - np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)]),
    np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)]) - 1, # Necessary?
]
g_u = to_poly(g_u, variables)

g_init = [
    np.prod(Z[0] * sym.conjugate(Z[0]) - (1/N - 0.05)),
    np.prod(1/N + 0.05 - Z[0] * sym.conjugate(Z[0])),
    1 - np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)]),
    np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)]) - 1, # Necessary?
]
g_init = to_poly(g_init, variables)


g_inv = [
    1 - np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)]),
    np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)]) - 1, # Necessary?
]
g_inv = to_poly(g_inv, variables)

g = {}
g[UNSAFE] = g_u
g[INIT] = g_init
g[INVARIANT] = g_inv
print("g defined")
if verbose: print(g)


# 2. Generate lam, barrier
lams = {}
for key in g:
    lam = [create_polynomial(variables, deg=2, coeff_tok='s_' + key + str(i)+';') for i in range(len(g_u))]
    lams[key] = lam
lam_u = lams[UNSAFE]
print("lam defined")
if verbose: print(lams)

# Define barrier in sympy
barrier = create_polynomial(variables, deg=2, coeff_tok='b')
print("Barrier made")
if verbose: print(barrier)

# 3. Make arbitrary polynomials
polys = {}
for key in [INIT, UNSAFE, DIFF]:
    polys[key] = poly_eq[key](barrier, lams, g)
print("Polynomials made")
if verbose: print(polys)

lam_coeffs = {}
for key in lams: lam_coeffs[key] = flatten([[next(iter(coeff.free_symbols)) for coeff in lam.coeffs()] for lam in lams[key]])

barrier_coeffs = [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]

symbol_var_dict = {}
for lam_symbols in lam_coeffs.values():
    symbol_var_dict.update(symbols_to_cvx_var(lam_symbols))
symbol_var_dict.update(symbols_to_cvx_var(barrier_coeffs))

# Needed?
symbols = {}
symbols[UNSAFE] = lam_coeffs[UNSAFE] + barrier_coeffs
symbols[INIT] = lam_coeffs[INIT] + barrier_coeffs
symbols[DIFF] = lam_coeffs[INVARIANT] + barrier_coeffs

# 4. Get matrix polynomial and constraints
constraints = []
matrices = []

print("Getting lam constraints...")
for key in lams:
    for poly in lams[key]:
        S_CVX, lam_constraints = PSD_constraint_generator(poly, symbol_var_dict)
        matrices.append(S_CVX)
        constraints += lam_constraints
print("lam constraints generated.")

print("Generating polynomial constraints...")
for key in polys:
    Q_CVX, poly_constraint = PSD_constraint_generator(polys[key], symbol_var_dict)
    matrices.append(Q_CVX)
    constraints += poly_constraint
constraints += [M >> 0 for M in matrices]
print("Poly constraints generated.")

# 5. Solve using cvxpy
obj = cp.Minimize(0)
prob = cp.Problem(obj, constraints)
print("Solving problem...")
prob.solve()
print(prob.status)

# 6. Return it in a readable format
# TODO: work out why some return None
# TODO: Fix value association from symbols to cvx_vars
symbols = symbols[UNSAFE] + symbols[INIT] + symbols[DIFF]
symbols = list(set(symbols))
symbols.sort(key = lambda symbol: symbol.name)
symbol_values = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
print(symbol_values)
print("Barrier: ", barrier.subs(symbol_values))
for i in range(len(lam_u)): print(f"lam{i}: {lam_u[i].subs(symbol_values)}")