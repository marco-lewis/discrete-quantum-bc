from examples.run_example import run_example
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
k = 2

log_level=logging.INFO
file_tag = "xgate" + str(n) + "k" + str(k)
verbose = 1

xgate = np.array([[0,1],[1,0]])
unitary = xgate
for i in range(1, n): unitary = np.kron(unitary, xgate)
circuit = [unitary] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = poly_list(g_inv, variables)
# Might need larger delta value
experiment = 1
g_u = [Z[experiment] * sym.conjugate(Z[experiment]) - 0.6]
g_u = poly_list(g_u, variables)

g_init = [
    Z[0] * sym.conjugate(Z[0]) - 0.49,
    0.51 - Z[0] * sym.conjugate(Z[0]),
    1j*(Z[0] - sym.conjugate(Z[0])),
    -1j*(Z[0] - sym.conjugate(Z[0])),
    ]
g_init = poly_list(g_init, variables)

g = {}
g[UNSAFE] = g_u + g_inv
g[INIT] = g_init + g_inv
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, solver='cvxopt', check=True)