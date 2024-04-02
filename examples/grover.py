from examples.run_example import run_example
from examples.gates import *
from src.utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 2
mark = 2
N = 2**n
eps = 0.01
gamma = 0.01
barrier_degree = 2
k = 1

log_level=logging.INFO
file_tag = "grover" + str(n) + "_" + "m" + str(mark)
verbose = 1

oracle = np.eye(N, N)
oracle[mark, mark] = -1
diffusion_oracle = np.eye(N,N)
temp = np.zeros((N,N))
temp[0,0] = 1
diffusion_oracle = 2*temp - diffusion_oracle

diffusion = np.dot(n_gate(Hgate, n), np.dot(diffusion_oracle, n_gate(Hgate, n)))
faulty_grover = np.dot(diffusion, oracle)
circuit = [oracle, diffusion]

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

# Marked state will never reach close to 1 (>90%)
g_u = [
    Z[mark] * sym.conjugate(Z[mark]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = poly_list(g_u, variables)

# Start close to superposition
err = 10 ** -(n+1)
g_init = []
g_init += [z * sym.conjugate(z) - (1/N - err) for z in Z]
g_init += [(1/N + err) - z * sym.conjugate(z) for z in Z]
g_init += [-1j * (z - sym.conjugate(z)) for z in Z]
g_init += [ 1j * (z - sym.conjugate(z)) for z in Z]
g_init += [
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

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, check=True)