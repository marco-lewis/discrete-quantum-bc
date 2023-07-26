from utils import *

def direct_method(unitary, g, Z, eps=0.01, verbose=0,n=2):
    N = 2**n
    variables = Z + [z.conjugate() for z in Z]

    # 1. Generate lam, barrier
    lams = {}
    for key in g:
        lam = [create_polynomial(variables, deg=g[key][i].total_degree(), coeff_tok='s_' + key + str(i)+';') for i in range(len(g[key]))]
        lams[key] = lam
    print("lam defined")
    if verbose: print(lams)

    barrier = create_polynomial(variables, deg=2, coeff_tok='b')
    print("Barrier made")
    if verbose: print(barrier)

    # 2. Make arbitrary polynomials for SOS terms
    dot = lambda lam, g: np.sum([li * gi for li,gi in zip(lam, g)])
    sym_poly_eq = dict([
        (INIT,lambda B, lams, g: -B - dot(lams[INIT], g[INIT])),
        (UNSAFE,lambda B, lams, g: B - eps - dot(lams[UNSAFE], g[UNSAFE])),
        (DIFF,lambda dB, lams, g: -dB - dot(lams[INVARIANT], g[INVARIANT])),
        # (LOC,lambda B, lam, g: -B - dot(lam, g)),
        ])
    sym_polys = {}
    for key in [INIT, UNSAFE]:
        sym_polys[key] = sym_poly_eq[key](barrier, lams, g)
    sym_polys[DIFF] = sym_poly_eq[DIFF](barrier.subs(zip(Z, np.dot(unitary, Z))) - barrier, lams, g)
    print("Polynomials made")
    if verbose: print(sym_polys)

    lam_coeffs = {}
    for key in lams: lam_coeffs[key] = flatten([[next(iter(coeff.free_symbols)) for coeff in lam.coeffs()] for lam in lams[key]])

    barrier_coeffs = [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]

    symbol_var_dict = {}
    for lam_symbols in lam_coeffs.values():
        symbol_var_dict.update(symbols_to_cvx_var_dict(lam_symbols))
    symbol_var_dict.update(symbols_to_cvx_var_dict(barrier_coeffs))

    # 3. Get matrix polynomial and constraints
    cvx_constraints = []
    cvx_matrices = []

    print("Getting lam constraints...")
    for key in lams:
        for poly in lams[key]:
            S_CVX, lam_constraints = PSD_constraint_generator(poly, symbol_var_dict, matrix_name='LAM_' + str(key), variables=variables)
            cvx_matrices.append(S_CVX)
            cvx_constraints += lam_constraints
    print("lam constraints generated.")

    print("Generating polynomial constraints...")
    for key in sym_polys:
        Q_CVX, poly_constraint = PSD_constraint_generator(sym_polys[key], symbol_var_dict, matrix_name='POLY_' + str(key), variables=variables)
        cvx_matrices.append(Q_CVX)
        cvx_constraints += poly_constraint
    cvx_constraints += [M >> 0 for M in cvx_matrices]
    print("Poly constraints generated.")

    # 4. Solve using cvxpy
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, cvx_constraints)
    print("Solving problem...")
    prob.solve()
    print(prob.status)

    # 5. Print the barrier in a readable format
    symbols = lam_coeffs[INIT] + lam_coeffs[UNSAFE] + lam_coeffs[INVARIANT] + barrier_coeffs
    symbols = list(set(symbols))
    symbols.sort(key = lambda symbol: symbol.name)
    symbol_values = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
    return barrier.subs(symbol_values)