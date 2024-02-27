from examples.run_example import run_example
from examples.gates import HNgate
from src.utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 1
N = 2**n
eps = 0.1
gamma = 0
barrier_degree = 2
k = 1

log_level=logging.INFO
file_tag = "hadamard" + str(n) + "k" + str(k)
verbose = 1

unitary = HNgate(n)
circuit = [unitary] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [Z[1] * sym.conjugate(Z[1]) - 0.9]
g_u = poly_list(g_u, variables)

g_init = [Z[0] * sym.conjugate(Z[0]) - 0.99]
g_init = poly_list(g_init, variables)

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = poly_list(g_inv, variables)

g = {}
g[UNSAFE] = g_u + g_inv
g[INIT] = g_init + g_inv
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, check=True)