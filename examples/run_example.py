from src.find_barrier_certificate import find_barrier_certificate
from src.log_settings import setup_logger
from src.typings import *
from src.utils import *
import examples.examples as ex

import argparse
import datetime
import logging

def add_invariant(g : SemiAlgebraicDict, Z : list[sym.Symbol], variables : list[sym.Symbol], n : int):
    sum_probs = np.sum([Z[j] * sym.conjugate(Z[j]) for j in range(2**n)])
    g_inv = [
        1 - sum_probs,
        sum_probs - 1,
    ]
    g_inv = poly_list(g_inv, variables)
    g[INVARIANT] = g_inv
    g[UNSAFE] += g_inv
    g[INIT] += g_inv
    return g

def run_example(file_tag : str, 
                circuit : Circuit,
                g : SemiAlgebraicDict,
                Z : list[sym.Symbol],
                barrier_degree=2,
                epsilon=0.01,
                gamma=0.01,
                k=2,
                verbose=1,
                log_level=logging.INFO,
                precision_bound=1e-4,
                solver='cvxopt',
                check=False):
    logger = setup_logger(file_tag + ".log", log_level=log_level)
    try:
        logger.info(str(datetime.datetime.now()))
        logger.info("g defined")
        logger.debug(g)
        
        barrier_certificate, times = find_barrier_certificate(
            circuit, g, Z, barrier_degree=barrier_degree, eps=epsilon,
            gamma=gamma, k=k, verbose=verbose, log_level=log_level,
            precision_bound=precision_bound, solver=solver, check=check
            )
        logger.info("Barrier certificate: " +  str(barrier_certificate))
        with open("logs/barrier_" + file_tag + ".log", 'w') as file:
            file.write("\n".join([
                str(datetime.datetime.now()),
                "k: " + str(k) + "; eps: " + str(epsilon) + "; gamma: " + str(gamma),
                repr(barrier_certificate)
            ]))
        logger.info("Barriers stored")
        return times
    except KeyboardInterrupt as e:
        logger.exception(e)

parser = argparse.ArgumentParser(
    description="Run an example.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
examples = ['xgate','zgate','xzgate','grover_even_unmark','grover_odd_unmark']
parser.add_argument("--example", "-ex", type=str, choices=examples, required=True, help="Example to run.")
parser.add_argument("--runs", type=int, default=1, help="Number of runs to perform.")
parser.add_argument("--solver", type=str, default='cvxopt', choices=['cvxopt', 'conelp', 'mosek'], help="SDP solver to use.")
parser.add_argument("-n", type=int, default=1, help="Number of qubits.")
parser.add_argument("--epsilon", "-eps", type=float, default=0.01, metavar="EPS", help="Bound for difference condition (B_t(f_t(x)) - B_t(x) < EPS).")
parser.add_argument("--gamma", "-gam", type=float, default=0.01, metavar="GAM", help="Bound for change condition (B_{t+1}(x) - B_t(x) < GAM).")
parser.add_argument("-k", type=int, default=1, help="Inductive parameter.")
parser.add_argument("--barrier-degree", type=int, default=2, help="Maximum degree of generated barrier.")
parser.add_argument("--target", "-tgt", type=int, default=0, help="Target qubit for semialgebraic sets.")
parser.add_argument("--mark", type=int, default=0, help="Marked qubit value for Grover example.")
parser.add_argument("--verbose", "-v", action='store_true', help="Set verbosity.")
parser.add_argument("--log-level", type=str, default='INFO', choices=["DEBUG,INFO,WARN,ERROR,CRITICAL"], help="Display level of logging.")
parser.add_argument("--check", action='store_true', help="Set to check generated barrier with SMT solvers.")

if __name__ == '__main__':
    args = parser.parse_args()
    Z = [sym.Symbol('z' + str(i), complex=True) for i in range(2**args.n)]
    variables = Z + [z.conjugate() for z in Z]
    if args.example == 'zgate':
        file_tag, circuit, g = ex.Z_example(Z, variables, args.n, args.k, args.target)
    if args.example == 'xgate':
        file_tag, circuit, g = ex.X_example(Z, variables, args.n, args.k, args.target)
    if args.example == 'xzgate':
        file_tag, circuit, g = ex.XZ_example(Z, variables, args.n, args.k, args.target)
    if args.example in ['grover_even_unmark', 'grover_odd_unmark']:
        file_tag, circuit, g = ex.Grover_unmark_example(Z, variables, args.n, args.k, args.target, args.mark, odd='odd' in args.example)
    g = add_invariant(g, Z, variables, args.n)

    run_times = {
        TIME_SP: [],
        TIME_PICOS: [],
        TIME_VERIF: []
    }
    for i in range(1, args.runs+1):
        print("Run " + str(i))
        times = run_example(
            file_tag=file_tag,
            circuit=circuit,
            g=g,
            Z=Z,
            barrier_degree=args.barrier_degree,
            epsilon=args.epsilon,
            gamma=args.gamma,
            k=args.k,
            verbose=args.verbose,
            log_level=args.log_level,
            solver=args.solver,
            check=args.check
            )
        for key in run_times: run_times[key].append(times[key])

    print(row_msg("Process", "Average times"))
    average = lambda l: sum(l)/len(l) if l != 0 else 0
    average_time = {}
    for key in run_times:
        average_time[key] = average(run_times[key])
        print(row_msg(key, format_time(average_time[key])))
    with open(f"logs/times/{file_tag}.log", 'w') as file: file.write(f"Run Time\n{run_times}\nAverage Time\n{average_time}")