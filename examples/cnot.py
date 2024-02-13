from examples.run_example import run_example
from src.utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 2
N = 2**n
eps = 0.01
gamma = 0.01
barrier_degree = 2
k = 1

log_level=logging.INFO
file_tag = "cnot_k" + str(k)
verbose = 1

cnot = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,0,1.0],[0,0,1.0,0]])
circuit = [cnot] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [
    Z[1] * sym.conjugate(Z[1]) + Z[0] * sym.conjugate(Z[0]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = poly_list(g_u, variables)

g_init = [
    Z[3] * sym.conjugate(Z[3]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
    ]
g_init = poly_list(g_init, variables)

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = poly_list(g_inv, variables)

g = {}
g[UNSAFE] = g_u
g[INIT] = g_init
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, solver='mosek', check=True)