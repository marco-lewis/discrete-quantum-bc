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

hadamard = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
unitary = hadamard
for i in range(1, n): unitary = np.kron(unitary, hadamard)
circuit = [unitary] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [
    Z[1] * sym.conjugate(Z[1]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = poly_list(g_u, variables)

g_init = []
g_init += [
    Z[0] * sym.conjugate(Z[0]) - 0.99,
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