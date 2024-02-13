from src.typings import *

from collections import defaultdict
from functools import reduce
import itertools
import logging

import numpy as np
import picos
import sympy as sym
from sympy.matrices.expressions.matexpr import MatrixElement

logger = logging.getLogger("utils")

DREAL = "dreal"
DREAL_SAT = "delta-sat"
DREAL_UNSAT = "unsat"
DREAL_UNKOWN = "unkown"
Z3 = "Z3"

UNSAFE = 'unsafe'
INIT = 'init'
INVARIANT = 'invariant'
BARRIER = 'barrier'
DIFF = 'diff'
LOC = 'loc'
INDUCTIVE = 'inductive'
CHANGE = 'change'

def flatten(matrix): return [item for row in matrix for item in row]

def calculate_d(k = 1, eps = 0.01, gamma = 0.01):
    return (k + 1) * (eps + gamma)

def generate_unitary_k(k : int, unitary : Unitary):
    unitary_k = unitary
    for i in range(1,k): unitary_k = np.dot(unitary, unitary_k)
    return unitary_k

def generate_variables(Z : list[sym.Symbol]):
    return Z + [z.conjugate() for z in Z]

def to_poly(expr_list : list[sym.Poly], variables : list[sym.Symbol], domain=sym.CC):
    return [sym.poly(l, variables, domain=domain) for l in expr_list]

def create_polynomial(variables : list[sym.Symbol], deg=2, coeff_tok='a', monomial=False) -> sym.Poly:
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

def convert_exprs(exprs : list[sym.Poly], symbol_var_dict : dict[sym.Symbol, picos.ComplexVariable]):
    def convert(expr):
        if isinstance(expr, sym.Add): return picos.sum([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Mul): return reduce(lambda x, y: x * y, [convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Pow): return convert(expr.args[0])**convert(expr.args[1])
        if symbol_var_dict.get(expr) is not None: return symbol_var_dict[expr]
        return complex(expr)
    return [convert(expr) for expr in exprs]

def convert_exprs_of_matrix(exprs : list[sym.Poly], cvx_matrix : picos.HermitianVariable):
    def convert(expr):
        if isinstance(expr, sym.Add): return picos.sum([convert(arg) for arg in expr.args])
        if isinstance(expr, sym.Mul): return reduce(lambda x, y: x * y, [convert(arg) for arg in expr.args], 1)
        if isinstance(expr, MatrixElement): return cvx_matrix[int(expr.i), int(expr.j)]
        return complex(expr)
    return [convert(expr) for expr in exprs]