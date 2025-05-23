from examples.gates import *
from src.utils import *

import numpy as np
import sympy as sym

def Z_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1, k=1, target=0):
    file_tag = f"zgate{n}_k{k}_tgt{target}"

    unitary = n_gate(Zgate, n)
    circuit = [unitary] * 6

    g_init = [
        Z[target] * sym.conjugate(Z[target]) - 0.9,
        1j*(Z[0] - sym.conjugate(Z[0])),
        -1j*(Z[0] - sym.conjugate(Z[0])),
        ]
    Z_no_target = Z[:target] + Z[target+1:]
    g_u = [z * sym.conjugate(z) - 0.2 for z in Z_no_target]
    g_init = poly_list(g_init, variables)
    g_u = poly_list(g_u, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def X_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1, k=1, target=0):
    N = 2**n
    file_tag = f"xgate{n}_k{k}_tgt{target}"

    circuit = [n_gate(Xgate, n)] * 6

    err = 10**-(n+1)
    g_u = [Z[target] * sym.conjugate(Z[target]) - (1/N + 2*err)]
    g_u = poly_list(g_u, variables)

    g_init = [z * sym.conjugate(z) - (1/N - err) for z in Z]
    g_init += [1/N + err - z * sym.conjugate(z) for z in Z]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

# target = 0, 1 good for n=1
# n=2, experiment = 0 good; = 1, 2, 3 bad
def XZ_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1, k=1, target=0):
    file_tag = f"xz{n}_k{k}_tgt{target}"

    circuit = [n_gate(Xgate, n), n_gate(Zgate, n)] * 6

    g_init = [Z[target] * sym.conjugate(Z[target]) - 0.9]
    g_u = [
        0.8 - Z[target] * sym.conjugate(Z[target]),
        Z[target] * sym.conjugate(Z[target]) - 0.2,
        ]
    g_init = poly_list(g_init, variables)
    g_u = poly_list(g_u, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def CX_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=2, k=1, ctrl_mark=0):
    if not(n == 2): raise Exception('Number of qubits needs to be even.')
    N = 2**n
    file_tag = f"cnot_k{k}_ctrl{ctrl_mark}"

    circuit = [CXgate] * 4

    modifier = 2 *(1 - ctrl_mark)
    g_u = [Z[modifier] * sym.conjugate(Z[modifier]) + Z[modifier + 1] * sym.conjugate(Z[modifier + 1]) - 0.9]
    g_u = poly_list(g_u, variables)

    g_init = [Z[2 * ctrl_mark] * sym.conjugate(Z[2 * ctrl_mark]) - 0.9]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def CZ_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=2, k=1):
    if not(n == 2): raise Exception('Number of qubits needs to be even.')
    N = 2**n
    file_tag = f"cz_k{k}"

    circuit = [CZgate] * 4

    g_u = [sum([z * sym.conjugate(z) for z in Z[:-1]]) - 0.2]
    g_u = poly_list(g_u, variables)

    g_init = [Z[3] * sym.conjugate(Z[3]) - 0.9]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def CH_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=2, k=1):
    if not(n == 2): raise Exception('Number of qubits needs to be even.')
    N = 2**n
    file_tag = f"ch_k{k}"

    circuit = [CHgate] * 4

    g_u = [sum([z * sym.conjugate(z) for z in Z[:-2]]) - 0.2]
    g_u = poly_list(g_u, variables)

    g_init = [Z[3] * sym.conjugate(Z[3]) - 0.9]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def CCX_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=2, k=1):
    if not(n == 3): raise Exception('Number of qubits (n) needs to be 3.')
    N = 2**n
    file_tag = f"ccx_k{k}"

    circuit = [CCXgate] * 4

    g_u = [sum([z * sym.conjugate(z) for z in Z[:-2]]) - 0.2]
    g_u = poly_list(g_u, variables)

    g_init = [Z[-1] * sym.conjugate(Z[-1]) - 0.9]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def Grover_simple_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1, k=1, target=0, mark=1):
    N = 2**n
    file_tag = f"grover_simple{n}_k{k}_m{mark}_tgt{target}"
    if target == mark: raise Exception("target value needs to be different from mark value.")

    oracle = np.eye(N, N)
    oracle[mark, mark] = -1
    diffusion_oracle = np.eye(N,N)
    temp = np.zeros((N,N))
    temp[0,0] = 1
    diffusion_oracle = 2*temp - diffusion_oracle

    diffusion = np.dot(n_gate(Hgate, n), np.dot(diffusion_oracle, n_gate(Hgate, n)))
    circuit = [np.dot(diffusion, oracle)] * 4

    g_u = [Z[target] * sym.conjugate(Z[target]) - 0.9]
    g_u = poly_list(g_u, variables)

    g_init = []
    g_init += [(z * sym.conjugate(z)) - 1/N for z in Z]
    g_init += [1/N - (z * sym.conjugate(z)) for z in Z]
    g_init += [(z - sym.conjugate(z))/2j for z in Z]
    g_init += [- (z - sym.conjugate(z))/2j for z in Z]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def Grover_dual_unmark_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1, k=1, target=0, mark=1):
    N = 2**n
    file_tag = f"grover_dual{n}_k{k}_m{mark}_tgt{target}"
    if target == mark: raise Exception("target value needs to be different from mark value.")

    oracle = np.eye(N, N)
    oracle[mark, mark] = -1
    diffusion_oracle = np.eye(N,N)
    temp = np.zeros((N,N))
    temp[0,0] = 1
    diffusion_oracle = 2*temp - diffusion_oracle

    diffusion = np.dot(n_gate(Hgate, n), np.dot(diffusion_oracle, n_gate(Hgate, n)))
    circuit = [oracle, diffusion] * 4

    g_u = [Z[target] * sym.conjugate(Z[target]) - 0.9]
    g_u = poly_list(g_u, variables)

    err = 10 ** -(n+1)
    g_init = []
    g_init += [(z * sym.conjugate(z)) - (1/N - err) for z in Z]
    g_init += [1/N + err - (z * sym.conjugate(z)) for z in Z]
    g_init += [(z - sym.conjugate(z))/2j + np.sqrt(err) for z in Z]
    g_init += [np.sqrt(err) - (z - sym.conjugate(z))/2j for z in Z]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g

def Grover_unmark_example(Z : list[sym.Symbol], variables : list[sym.Symbol], n=1, k=1, target=0, mark=1, odd=0):
    N = 2**n
    odd_str = "odd" if odd else "even"
    file_tag = f"grover_unmark{n}_{odd_str}_k{k}_m{mark}_tgt{target}"
    if target == mark: raise Exception("target value needs to be different from mark value.")

    oracle = np.eye(N, N)
    oracle[mark, mark] = -1
    diffusion_oracle = np.eye(N,N)
    temp = np.zeros((N,N))
    temp[0,0] = 1
    diffusion_oracle = 2*temp - diffusion_oracle

    diffusion = np.dot(n_gate(Hgate, n), np.dot(diffusion_oracle, n_gate(Hgate, n)))
    circuit = [np.dot(oracle, diffusion)] * 4 if odd else [np.dot(diffusion, oracle)] * 4

    g_u = [Z[target] * sym.conjugate(Z[target]) - 0.9]
    g_u = poly_list(g_u, variables)

    err = 10 ** -(n+1)
    g_init = []
    g_init += [(z * sym.conjugate(z)) - (1/N - err) for z in Z]
    g_init += [1/N + err - (z * sym.conjugate(z)) for z in Z]
    g_init += [(z - sym.conjugate(z))/2j + np.sqrt(err) for z in Z]
    g_init += [np.sqrt(err) - (z - sym.conjugate(z))/2j for z in Z]
    if odd: g_init = [g.subs(zip(Z, np.dot(oracle, Z))) for g in g_init]
    g_init = poly_list(g_init, variables)

    g = {}
    g[UNSAFE] = g_u
    g[INIT] = g_init
    return file_tag, circuit, g