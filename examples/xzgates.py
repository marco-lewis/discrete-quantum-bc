from examples.run_example import run_example
from examples.gates import *
from src.utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 2
N = 2**n
eps = 0.1
gamma = 0.1
barrier_degree = 2
k = 2

log_level=logging.INFO
file_tag = "x_z" + str(n) + "k" + str(k)
verbose = 1

u_x = lambda n: Xgate if n == 1 else np.kron(Xgate, u_x(n-1))
u_z = lambda n: Zgate if n == 1 else np.kron(Zgate, u_z(n-1))
circuit = [u_x(n), u_z(n)] * 6

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = poly_list(g_inv, variables)

# experiment = 0, 1 good for n=1
# n=2, experiment = 0 good, = 1, 3 bad
experiment = 0
g_init = [
    Z[experiment] * sym.conjugate(Z[experiment]) - 0.9,
    ]
g_u = [
    0.8 - Z[experiment] * sym.conjugate(Z[experiment]),
    Z[experiment] * sym.conjugate(Z[experiment]) - 0.2,
    ]
g_init = poly_list(g_init, variables)
g_u = poly_list(g_u, variables)

g = {}
g[UNSAFE] = g_u + g_inv
g[INIT] = g_init + g_inv
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, solver='cvxopt', check=True)