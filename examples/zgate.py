from examples.gates import n_gate, Zgate
from src.utils import *

import numpy as np
import sympy as sym

def Zgate_experiment(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1,k=1,target=0):
    N = 2**n
    file_tag = "zgate" + str(n) + "k" + str(k)

    unitary = n_gate(Zgate, n)
    circuit = [unitary] * 6

    g_init = [
        Z[target] * sym.conjugate(Z[target]) - 0.9,
        1j*(Z[0] - sym.conjugate(Z[0])),
        -1j*(Z[0] - sym.conjugate(Z[0])),
        ]
    g_u = [
        Z[1 - target] * sym.conjugate(Z[1 - target]) - 0.2,
        ]
    g_init = poly_list(g_init, variables)
    g_u = poly_list(g_u, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g