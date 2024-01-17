from collections import defaultdict
from functools import reduce
import itertools
import logging
from typing import Dict, List

import numpy as np
import picos
import sympy as sym
from sympy.matrices.expressions.matexpr import MatrixElement

logger = logging.getLogger("utils")

UNSAFE = 'unsafe'
INIT = 'init'
INVARIANT = 'invariant'
BARRIER = 'barrier'
DIFF = 'diff'
LOC = 'loc'
INDUCTIVE = 'inductive'
CHANGE = 'change'

def flatten(matrix): return [item for row in matrix for item in row]

def calculate_d(k = 1, eps = 0.01, gamma = 0):
    return (k + 1) * (eps + gamma)

def generate_unitary_k(k, unitary):
    unitary_k = unitary
    for i in range(1,k): unitary_k = np.dot(unitary, unitary_k)
    return unitary_k

def to_poly(expr_list, variables, domain=sym.CC):
    return [sym.poly(l, variables, domain=domain) for l in expr_list]

def create_polynomial(variables, deg=2, coeff_tok='a', monomial=False) -> sym.Poly :
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

def symbols_to_cvx_var_dict(symbols : List[sym.Symbol]):
    cvx_vars = [picos.ComplexVariable(name = s.name) for s in symbols]
    symbol_var_dict = dict(zip(symbols, cvx_vars))
    return symbol_var_dict

def convert_exprs(exprs : List[sym.Poly], symbol_var_dict : Dict[sym.Symbol, picos.ComplexVariable]):
    def convert(expr):
        if isinstance(expr, sym.Add): return picos.sum([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Mul): return reduce(lambda x, y: x * y, [convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Pow): return convert(expr.args[0])**convert(expr.args[1])
        if symbol_var_dict.get(expr) is not None: return symbol_var_dict[expr]
        return complex(expr)
    return [convert(expr) for expr in exprs]

def convert_exprs_of_matrix(exprs : List[sym.Poly], cvx_matrix : picos.HermitianVariable):
    def convert(expr):
        if isinstance(expr, sym.Add): return picos.sum([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Mul): return reduce(lambda x, y: x * y, [convert(arg) for arg in expr.args], 1)
        if isinstance(expr, MatrixElement): return cvx_matrix[int(expr.i), int(expr.j)]
        return complex(expr)
    return [convert(expr) for expr in exprs]

def PSD_constraint_generator(sym_polynomial : sym.Poly,
                             symbol_var_dict : Dict[sym.Symbol, picos.ComplexVariable],
                             matrix_name='Q',
                             variables=[]):
    # Setup dictionary of monomials to cvx coefficients for sym_polynomial
    cvx_coeffs = convert_exprs(sym_polynomial.coeffs(), symbol_var_dict)
    poly_monom_to_cvx = dict(zip(sym_polynomial.monoms(), cvx_coeffs))
    poly_monom_to_cvx = defaultdict(lambda: 0.0, poly_monom_to_cvx)

    # Create sympy matrix and quadratic form as polynomial
    m = create_polynomial(variables[:len(variables)//2], deg=sym_polynomial.total_degree()//2, monomial=True)
    vector_monomials = np.array([np.prod([x**k for x, k in zip(m.gens, mon)]) for mon in m.monoms()])
    num_of_monom = len(vector_monomials)
    Q_SYM = sym.MatrixSymbol(matrix_name, num_of_monom, num_of_monom)
    vH_Q_v = sym.poly(vector_monomials.conj().T @ Q_SYM @ vector_monomials, variables)

    # Create cvx matrix and dictionary of monomials to cvx matrix terms
    Q_CVX = picos.HermitianVariable(name=matrix_name, shape=(num_of_monom, num_of_monom))
    Q_cvx_coeffs = convert_exprs_of_matrix(vH_Q_v.coeffs(), Q_CVX)
    Q_monom_to_cvx = dict(zip(vH_Q_v.monoms(), Q_cvx_coeffs))

    # Link matrix variables to polynomial variables
    constraints = [Q_monom_to_cvx[key] == poly_monom_to_cvx[key] for key in Q_monom_to_cvx]
    return Q_CVX, constraints