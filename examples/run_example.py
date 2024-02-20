from src.find_barrier_certificate import find_barrier_certificate
from src.log_settings import setup_logger
from src.typings import *

import datetime
import logging

def run_example(file_tag : str, 
                circuit : Circuit,
                g : SemiAlgebraicDict,
                Z : list[sym.Symbol],
                barrier_degree=2,
                eps=0.01,
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
        
        barrier_certificate = find_barrier_certificate(circuit, g, Z, barrier_degree=barrier_degree, eps=eps, gamma=gamma, k=k, verbose=verbose, log_level=log_level, precision_bound=precision_bound, solver=solver, check=check)
        logger.info("Barrier certificate: " +  str(barrier_certificate))
        with open("logs/barrier_" + file_tag + ".log", 'w') as file:
            file.write("k: " + str(k) + "; eps: " + str(eps) + "; gamma: " + str(gamma) + "\n")
            file.write(repr(barrier_certificate))
        logger.info("Barriers stored")
    except KeyboardInterrupt as e:
        logger.exception(e)