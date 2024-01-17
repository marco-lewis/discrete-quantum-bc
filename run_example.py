from direct import direct_method
from log_settings import setup_logger

import logging

def run_example(file_tag, circuit, g, Z, barrier_degree=2, eps=0.01, k=2, verbose=1, log_level=logging.INFO, precision_bound=1e-4, solver='cvxopt'):
    logger = setup_logger(file_tag + ".log", log_level=log_level)
    logger.info("g defined")
    logger.debug(g)

    barrier = direct_method(circuit, g, Z, barrier_degree=barrier_degree, eps=eps, k=k, verbose=verbose, log_level=log_level, precision_bound=precision_bound, solver=solver)
    logger.info("Barriers: " +  str(barrier))
    with open("logs/barrier_" + file_tag + ".log", 'w') as file:
        file.write(repr(barrier))
    logger.info("Barriers stored")