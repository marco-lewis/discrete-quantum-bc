from check import check_barrier
from direct import direct_method
from log_settings import setup_logger
from utils import *

import logging
import os

import numpy as np
import sympy as sym


log_level=logging.INFO
logger = setup_logger("grover.log", log_level=log_level)
verbose = 1

n = 1
N = 2**n

# 0. Inputs, variable definitions and constants
hadamard = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
unitary = hadamard

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [
    Z[1] * sym.conjugate(Z[1]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = to_poly(g_u, variables)

g_init = []
g_init += [
    Z[0] * sym.conjugate(Z[0]) - 0.9,
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

eps = 0.01
barrier_degree = 2
k = 2

barrier = direct_method(unitary, g, Z, barrier_degree=barrier_degree, eps=eps, k=k, verbose=verbose, log_level=log_level)
logger.info("Barrier: " +  str(barrier))
with open("logs/barrier_" + os.path.basename(__file__)[:-3] + ".log", 'w') as file:
    file.write(repr(barrier))
logger.info("Barrier stored")
if not(barrier == sym.core.numbers.Infinity): check_barrier(barrier, g, Z=Z, unitary=unitary, k=k, eps=eps, log_level=log_level)