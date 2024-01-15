from direct import direct_method
from log_settings import setup_logger
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
file_tag = "grover" + str(n) + "_" + "m" + str(mark)
logger = setup_logger(file_tag + ".log", log_level=log_level)
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
g_u = to_poly(g_u, variables)

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
logger.info("g defined")
logger.debug(g)

barrier = direct_method(circuit, g, Z, barrier_degree=barrier_degree, eps=eps, k=k, verbose=verbose, log_level=log_level, precision_bound=1e-4, solver='cvxopt')
logger.info("Barriers: " +  str(barrier))
with open("logs/barrier_" + file_tag + ".log", 'w') as file:
    file.write(repr(barrier))
logger.info("Barriers stored")