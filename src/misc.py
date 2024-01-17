from src.complex import *
import numpy as np
import z3

A = np.array([[Complex('a' + str(i) + str(j)) for j in range(3)] for i in range(1)])

def conj(A): return [[A[i][j].conj() for j in range(len(A[i]))] for i in range(len(A))]
def conjtrans(A): return conj(A.transpose())

Q = np.array([
    [0,0,0],
    [0,1/4,-1/4],
    [0,-1/4,1/4],
    ])

print(A)
print(conjtrans(A))

s = z3.Solver()

[[s.add(Q[i][j] == np.dot(conjtrans(A), A)[i][j]) for j in range(3)] for i in range(3)]

# p = [[Bool('p' + str(i) + str(j)) for j in range(3)] for i in range(3)]
# [[s.assert_and_track(Q[i][j] == np.dot(conjtrans(A), A)[i][j],p[i][j]) for j in range(3)] for i in range(3)]

print(s)
sat = s.check()
print(sat)
if sat == z3.sat:
    m = s.model()
    print(m)
    print([[evaluate_cexpr(m, A[i][j]) for j in range(3)] for i in range(1)])
if sat == z3.unsat: print(s.unsat_core())