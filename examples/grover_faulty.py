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
eps = 0.1
gamma = 0.1
barrier_degree = 2
k = 2

log_level=logging.INFO
file_tag = "grover_faulty" + str(n) + "_" + "m" + str(mark)
verbose = 1

oracle = np.eye(N, N)
oracle[mark, mark] = -1
diffusion_oracle = np.eye(N,N)
temp = np.zeros((N,N))
temp[0,0] = 1
diffusion_oracle = 2*temp - diffusion_oracle

Hgate = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
HNgate = lambda n: Hgate if n == 1 else np.kron(Hgate, HNgate(n-1))
diffusion_faulty = np.dot(np.kron(HNgate(n-1),Ygate), np.dot(diffusion_oracle, HNgate(n)))
faulty_grover = np.dot(diffusion_faulty, oracle)
circuit = [oracle, diffusion_faulty] * 2

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_inv = [
    1 - sum_probs,
    sum_probs - 1,
]
g_inv = poly_list(g_inv, variables)

# Marked state will never reach close to 1 (>90%)
g_u = [Z[mark] * sym.conjugate(Z[mark]) - 0.9]
g_u = poly_list(g_u, variables)

# Start close to superposition
err = 10 ** -(n+1)
g_init = []
g_init += [(z + sym.conjugate(z))/2 - np.sqrt(1/N - err) for z in Z]
g_init += [np.sqrt(1/N + err) - (z + sym.conjugate(z))/2 for z in Z]
g_init += [1 - np.sum([((z + sym.conjugate(z))/2)**2 for z in Z])]
g_init += [np.sum([((z + sym.conjugate(z))/2)**2 for z in Z]) - 1]
g_init = poly_list(g_init, variables)

g = {}
g[UNSAFE] = g_u + g_inv
g[INIT] = g_init
g[INVARIANT] = g_inv

run_example(file_tag, circuit, g, Z, barrier_degree, eps, gamma, k, verbose, log_level, check=True)