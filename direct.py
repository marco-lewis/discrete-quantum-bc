from utils import *

def direct_method(unitary : np.ndarray,
                  g : Dict[str, List[sym.Poly]],
                  Z : List[sym.Symbol],
                  barrier_degree=2,
                  eps=0.01,
                  verbose=0):
    variables = Z + [z.conjugate() for z in Z]

    # 1. Generate lam, barrier
    lams : Dict(str, sym.Poly) = {}
    for key in g:
        lams[key] = [create_polynomial(variables, deg=g[key][i].total_degree(), coeff_tok='s_' + key + str(i)+';') for i in range(len(g[key]))]
        print("lam polynomial for " + key + " made.")
    print("lams defined.")
    if verbose: print(lams)

    barrier = create_polynomial(variables, deg=barrier_degree, coeff_tok='b')
    print("Barrier made.")
    if verbose: print(barrier)

    # 2. Make arbitrary polynomials for SOS terms
    dot = lambda lam, g: np.sum([li * gi for li,gi in zip(lam, g)])
    sym_poly_eq = dict([
        (INIT,lambda B, lams, g: sym.poly(-B - dot(lams[INIT], g[INIT]), variables)),
        (UNSAFE,lambda B, lams, g: sym.poly(B - eps - dot(lams[UNSAFE], g[UNSAFE]), variables)),
        # (DIFF,lambda dB, lams, g: -dB),
        (DIFF,lambda B, lams, g: sym.poly(-B.subs(zip(Z, np.dot(unitary, Z))) + B - dot(lams[INVARIANT], g[INVARIANT]), variables)),
        # (LOC,lambda B, lam, g: -B - dot(lam, g)),
        ])
    sym_polys = {}
    for key in sym_poly_eq:
        sym_polys[key] = sym_poly_eq[key](barrier, lams, g)
        print("Polynomial for " + key + " made.")
    print("Polynomials made.")
    if verbose: print(sym_polys)

    lam_coeffs : Dict[str, List(sym.Symbol)] = {}
    for key in lams: lam_coeffs[key] = flatten([[next(iter(coeff.free_symbols)) for coeff in lam.coeffs()] for lam in lams[key]])

    barrier_coeffs = [next(iter(coeff.free_symbols)) for coeff in barrier.coeffs()]

    symbol_var_dict : Dict[sym.Symbol, cp.Variable]= {}
    for lam_symbols in lam_coeffs.values(): symbol_var_dict.update(symbols_to_cvx_var_dict(lam_symbols))
    symbol_var_dict.update(symbols_to_cvx_var_dict(barrier_coeffs))

    # 3. Get matrix polynomial and constraints
    cvx_constraints = []
    cvx_matrices : List[cp.Variable] = []

    print("Generating lam constraints...")
    for key in lams:
        i = 0
        for poly in lams[key]:
            S_CVX, lam_constraints = PSD_constraint_generator(poly, symbol_var_dict, matrix_name='LAM_' + str(key) + str(i), variables=variables)
            cvx_matrices.append(S_CVX)
            cvx_constraints += lam_constraints
            print(str(key) + str(i) + " done.")
            i += 1
    print("lam constraints generated.")

    print("Generating polynomial constraints...")
    for key in sym_polys:
        Q_CVX, poly_constraint = PSD_constraint_generator(sym_polys[key], symbol_var_dict, matrix_name='POLY_' + str(key), variables=variables)
        cvx_matrices.append(Q_CVX)
        cvx_constraints += poly_constraint
        print(str(key) + " done.")
    print("Poly constraints generated.")

    print("Generating semidefinite constraints...")
    cvx_constraints += [M >> 0 for M in cvx_matrices]
    print("Semidefinite constraints generated.")
    print("Constraints generated")
    if verbose: print(cvx_constraints)

    # 4. Solve using cvxpy
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, cvx_constraints)
    print("Solving problem...")
    prob.solve(verbose=bool(verbose))
    print(prob.status)
    if prob.status in ["infeasible", "unbounded"]: raise Exception("Cannot get barrier from problem.")

    # 5. Print the barrier in a readable format
    print("Fetching values...")
    symbols : List[sym.Symbol] = barrier_coeffs + lam_coeffs[INIT] + lam_coeffs[UNSAFE] + lam_coeffs[INVARIANT]
    symbols = list(set(symbols))
    symbols.sort(key = lambda symbol: symbol.name)
    symbol_values = dict(zip(symbols, [symbol_var_dict[s].value for s in symbols]))
    for key in symbol_values: symbol_values[key] = symbol_values[key] if abs(symbol_values[key]) > 1e-10 else 0
    if verbose:
        for key in lams:
            i = 0
            for lam in lams[key]:
                print("lam_" + str(key) + str(i), lam.subs(symbol_values))
                i += 1
        for m in cvx_matrices: print(m, m.value)
    return barrier.subs(symbol_values)