from run_example import run_example
from utils import *

import logging

import numpy as np
import sympy as sym

# 0. Inputs, variable definitions and constants
n = 2
mark = 2
N = 2**n
eps = 0.01
barrier_degree = 2
k = 1

log_level=logging.INFO
file_tag = "grover_unmark" + str(n) + "_" + "m" + str(mark)
verbose = 1

oracle = np.eye(N, N)
oracle[mark, mark] = -1
diffusion_oracle = np.eye(N,N)
temp = np.zeros((N,N))
temp[0,0] = 1
diffusion_oracle = 2*temp - diffusion_oracle

hadamard = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
hadamard_n = lambda n: hadamard if n == 1 else np.kron(hadamard, hadamard_n(n-1))
diffusion = np.dot(hadamard_n(n), np.dot(diffusion_oracle, hadamard_n(n)))
circuit = [oracle, diffusion]

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

# Unmarked states will never be likely (>50%)
g_u = [
    - np.prod([(Z[i] * sym.conjugate(Z[i]) - 0.5) if i != mark else 1 for i in range(N)]),
    1 - sum_probs,
    sum_probs - 1,
]
g_u = to_poly(g_u, variables)

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
g_init = to_poly(g_init, variables)

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = to_poly(g_inv, variables)

g = {}
g[UNSAFE] = g_u
g[INIT] = g_init
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, k, verbose, log_level)