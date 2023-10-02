from check import check_barrier
from direct import direct_method
from log_settings import setup_logger
from utils import *

import numpy as np
import sympy as sym

import logging

log_level=logging.INFO
logger = setup_logger("grover.log", log_level=log_level)
verbose = 1

n = 2
mark = 2
N = 2**n

# 0. Inputs, variable definitions and constants
oracle = np.eye(N, N)
oracle[mark, mark] = -1
diffusion_oracle = np.eye(N,N)
temp = np.zeros((N,N))
temp[0,0] = 1
diffusion_oracle = 2*temp - diffusion_oracle

hadamard = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
hadamard_n = lambda n: hadamard if n == 1 else np.kron(hadamard, hadamard_n(n-1))
diffusion = np.dot(hadamard_n(n), np.dot(diffusion_oracle, hadamard_n(n)))
grover = np.dot(diffusion, oracle)
faulty_grover = np.dot(np.dot(np.kron(hadamard_n(n-1),np.eye(2,2)), np.dot(diffusion_oracle, hadamard_n(n))), oracle)
unitary = faulty_grover

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [
    # Z[1] * sym.conjugate(Z[1]) - 0.9,
    # -np.prod([0.9 - Z[i] * sym.conjugate(Z[i]) if not(i == mark) else 1 for i in range(N)]),
    Z[mark] * sym.conjugate(Z[mark]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = to_poly(g_u, variables)

d = 0.01
g_init = []
g_init += [z * sym.conjugate(z) - (1/N - d) for z in Z]
g_init += [1/N + d - z * sym.conjugate(z) for z in Z]
# g_init += [-1j * (z - sym.conjugate(z)) for z in Z]
# g_init += [1j * (z - sym.conjugate(z)) for z in Z]
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

eps = 0.1
barrier_degree=2
k = 2

barrier = direct_method(unitary, g, Z, barrier_degree=barrier_degree, eps=eps, k=k, verbose=verbose, log_level=log_level)
logger.info("Barrier: " +  str(barrier))
with open("logs/barrier.log", 'w') as file:
    file.write(repr(barrier))
if not(barrier == sym.core.numbers.Infinity): check_barrier(barrier, g, Z=Z, unitary=unitary)