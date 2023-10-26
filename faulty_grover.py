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
k = 2

log_level=logging.INFO
logger = setup_logger("grover" + str(n) + "_" + "m" + str(mark) + ".log", log_level=log_level)
verbose = 1

oracle = np.eye(N, N)
oracle[mark, mark] = -1
diffusion_oracle = np.eye(N,N)
temp = np.zeros((N,N))
temp[0,0] = 1
diffusion_oracle = 2*temp - diffusion_oracle

hadamard = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
hadamard_n = lambda n: hadamard if n == 1 else np.kron(hadamard, hadamard_n(n-1))
faulty_grover = np.dot(np.dot(np.kron(hadamard_n(n-1),np.eye(2,2)), np.dot(diffusion_oracle, hadamard_n(n))), oracle)
circuit = [oracle, np.kron(hadamard_n(n-1),np.eye(2,2)), diffusion_oracle, hadamard_n(n)]

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [
    Z[mark] * sym.conjugate(Z[mark]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = to_poly(g_u, variables)

d = 0.01
g_init = [z * sym.conjugate(z) - (1/N - d) for z in Z]
g_init += [1/N + d - z * sym.conjugate(z) for z in Z]
g_init += [-1j * (z - sym.conjugate(z)) for z in Z]
g_init += [1j * (z - sym.conjugate(z)) for z in Z]
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

barrier = direct_method(circuit, g, Z, barrier_degree=barrier_degree, eps=eps, k=k, verbose=verbose, log_level=log_level)
logger.info("Barrier: " +  str(barrier))
with open("logs/barrier_" + __file__ + ".log", 'w') as file:
    file.write(repr(barrier))
logger.info("Barrier stored")