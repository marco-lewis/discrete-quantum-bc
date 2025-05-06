from examples.gates import *
from examples.run_example import add_invariant
from src.find_barrier_certificate import find_barrier_certificate
from src.utils import REACH, REACHINIT, poly_list

import logging
import sympy as sym

n = 1
Z = [sym.Symbol('z' + str(i), complex=True) for i in range(2**n)]
variables = Z + [z.conjugate() for z in Z]

H = -1j * Hgate
circuit = [H]
g_init = [
    Z[0] * sym.conjugate(Z[0]) - 0.9,
    1j*(Z[0] - sym.conjugate(Z[0])),
    -1j*(Z[0] - sym.conjugate(Z[0])),
    ]
s = 1e-5
g_reach = [
    Z[0] * sym.conjugate(Z[0]) - (0.5 - s),
    Z[1] * sym.conjugate(Z[1]) - (0.5 - s),
    (0.5 + s) - Z[0] * sym.conjugate(Z[0]),
    (0.5 + s) - Z[1] * sym.conjugate(Z[1]),
]

g_reach = [-g for g in g_reach]
g = {}
g[REACHINIT] = poly_list(g_init, variables)
g[REACH] = poly_list(g_reach, variables)
g = add_invariant(g, Z, variables, n)

barrier_degree=2
epsilon=0.01
gamma=0.01
k = 1
verbose = 1
log_level = logging.INFO
solver = 'cvxopt' if 1 else 'mosek'
smt_timeout = 60
docker = True

barrier_certificate, times = find_barrier_certificate(
            circuit, g, Z, barrier_degree=barrier_degree, eps=epsilon,
            gamma=gamma, k=k, verbose=verbose, log_level=log_level,
            solver=solver, smt_timeout=smt_timeout, isreach=2, docker=docker,
            )