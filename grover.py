from direct import direct_method
from utils import *

import numpy as np
import sympy as sym

verbose = 0

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

eps = 0.1

Z = [sym.Symbol('z' + str(i), complex=True) for i in range(N)]
variables = Z + [z.conjugate() for z in Z]

sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(N)])

g_u = [
    # -np.prod([0.7 - Z[i] * sym.conjugate(Z[i]) if not(i == mark) else 1 for i in range(N)]),
    Z[mark] * sym.conjugate(Z[mark]) - 0.9,
    1 - sum_probs,
    sum_probs - 1,
]
g_u = to_poly(g_u, variables)

g_init = [
    np.prod(Z[0] * sym.conjugate(Z[0]) - (1/N - 0.05)),
    np.prod(1/N + 0.05 - Z[0] * sym.conjugate(Z[0])),
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
print("g defined")
if verbose: print(g)

barrier = direct_method(unitary, g, Z,eps=eps)
print("Barrier: ", barrier)