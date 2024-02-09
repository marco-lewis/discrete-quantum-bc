from examples.run_example import run_example
from src.utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 2
N = 2**n
eps = 0.1
gamma = 0
barrier_degree = 2
k = 1

log_level=logging.INFO
file_tag = "hadamard" + str(n) + "k" + str(k)
verbose = 1

zgate = np.array([[1,0],[0,-1]])
unitary = zgate
for i in range(1, n): unitary = np.kron(unitary, zgate)
circuit = [unitary] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

experiment = 0
# experiment = 1
g_init = [
    Z[experiment] * sym.conjugate(Z[experiment]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
    ]
g_u = [
    Z[1 - experiment] * sym.conjugate(Z[1 - experiment]) - 0.2,
    1 - sum_probs,
    sum_probs - 1,
    ]
g_init = to_poly(g_init, variables)
g_u = to_poly(g_u, variables)

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = to_poly(g_inv, variables)

g = {}
g[UNSAFE] = g_u
g[INIT] = g_init
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, solver='mosek', check=True)