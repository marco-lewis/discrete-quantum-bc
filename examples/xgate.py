from examples.run_example import run_example
from examples.gates import Xgate
from src.utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 2
N = 2**n
eps = 0.01
gamma = 0
barrier_degree = 2
k = 2

log_level=logging.INFO
file_tag = "xgate" + str(n) + "k" + str(k)
verbose = 1

unitary = Xgate
for i in range(1, n): unitary = np.kron(unitary, Xgate)
circuit = [unitary] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = poly_list(g_inv, variables)

experiment = 0
err = 10**-(n+1)
g_u = [Z[experiment] * sym.conjugate(Z[experiment]) - (1/2**n + 2*err)]
g_u = poly_list(g_u, variables)

g_init = [
    1j*(Z[0] - sym.conjugate(Z[0])),
    -1j*(Z[0] - sym.conjugate(Z[0])),
    ]
g_init += [z * sym.conjugate(z) - (1/2**n - err) for z in Z]
g_init += [1/2**n + err - z * sym.conjugate(z) for z in Z]
g_init = poly_list(g_init, variables)

g = {}
g[UNSAFE] = g_u + g_inv
g[INIT] = g_init + g_inv
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, solver='cvxopt', check=True)