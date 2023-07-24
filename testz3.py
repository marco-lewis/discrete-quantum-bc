from complex import Complex
import numpy as np
from z3 import *

# set_option(verbose=10)
n = 1
track = 0
zs = [Complex('z' + str(i)) for i in range(2**n)]
vs = [Real('v' + str(i)) for i in range(2**n)]
c = Real('c')


abs_sqr = lambda zs: [z * z.conj() for z in zs]
list_complex = lambda zs: [z.r for z in zs] + [z.i for z in zs]
b = lambda vs, zs: sum([z * v for v, z in zip(vs, abs_sqr(zs))])
B = lambda zs: c + b(vs, zs).r
Bmodel = lambda c, vs, zs: c + b(vs, zs).r

sum_is_one = sum(abs_sqr(zs)) == 1

# TODO: Fix init and unsafe for Grover example
init = lambda zs: And(abs_sqr(zs)[0].r >= 0.9, sum_is_one)
unsafe = lambda zs: And(Or([z.r > 0.9 for z in abs_sqr(zs)][1:]), sum_is_one)

In = lambda n: np.identity(2**n)
Z = np.zeros((2**n, 2**n))
Z[0][0] = 2
s = 1/np.sqrt(2)
h = [[s,s],[s,-s]]
H = lambda n: h if n == 1 else np.dot(h, H(n-1))
D = lambda zs: np.dot(np.dot(H(2**n), np.dot(Z - In(n), H(n))), zs)

Ot = np.zeros((2**n, 2**n))
Ot[1][1] = 1
O = lambda zs: np.dot(In(n) - 2 * Ot, zs)
G = lambda zs: D(O(zs))

constraints = [
    ForAll(list_complex(zs), Implies(init(zs), B(zs) <= 0)),
    ForAll(list_complex(zs), Implies(unsafe(zs), B(zs) > 0)),
    # ForAll(list_complex(zs), And(Implies(init(zs), B(zs) <= 0), Implies(unsafe(zs), B(zs) > 0))),
    ForAll(list_complex(zs), Implies(sum_is_one, B(G(zs)) - B(zs) <= 0)),
    And([Not(v == 0) for v in vs]),
    # vs[0] == -1,
    # vs[1] == 1,
    # c == 0,
    ]

s = Solver()
s.set('v2', True)
trackers = [Bool('t' + str(i)) for i in range(len(constraints))]
if track: [s.assert_and_track(simplify(c),t) for c, t in zip(constraints, trackers)]
else: [s.add(simplify(c)) for c in constraints]
print(s.assertions())

sat = s.check()

print(sat)
if sat == z3.sat:
    print(s.model())
    print(simplify(Bmodel(s.model()[c], [s.model()[v] for v in vs], zs) == Real("B")))
elif sat == z3.unsat:
    print(s.unsat_core())