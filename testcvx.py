# Use discrete env
import cvxpy as cp
import numpy as np

n = 1
p = 2
mod1 = lambda x: cp.real(x)*cp.real(x) + cp.imag(x)*cp.imag(x)
mod2 = lambda x: x * cp.conj(x)
mod3 = lambda x: cp.square(cp.abs(x))
modl = lambda xs: [mod2(x) for x in xs]

zr = cp.Variable(2**n, name='zr')
zi = cp.Variable(2**n, name='zi')
z = zr + zi
z = cp.Variable(2**n, name='z', complex=True)

M = cp.Variable((2**n, 2**n),
                # complex=True,
                hermitian=True,
                )

constraints = [
    # cp.sum(modl(z)) == 1,
    # cp.square(cp.abs(z)) <= 1,
    # cp.sum_squares(z) == 1,
    M >> 0,
    ]

A = []
b = []
for i in range(p):
    A.append(np.random.randn(2**n, 2**n))
    b.append(np.random.randn())
constraints += [
    cp.trace(A[i] @ M) == b[i] for i in range(p)
]
print(constraints)

obj = cp.Minimize(0)

prob = cp.Problem(obj, constraints)
prob.solve(
    # verbose = True
    )
print(prob.status)
print(M.value)