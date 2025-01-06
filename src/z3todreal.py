from src.utils import *

import logging
import os
import subprocess
import sys

import z3

DREAL_PATH = '/opt/dreal/4.21.06.2/bin/dreal'

SMT2_PATH = 'logs/barrierforqcirc.smt2'
DOCKER_COMMAND = lambda delta: f"docker run -v .:/mnt --rm dreal/dreal4 dreal -j 15 /mnt/{SMT2_PATH} --precision {delta}"

logger = logging.getLogger("dreal")

def run_dreal(s : z3.Solver, delta=0.001, log_level=logging.INFO, timeout=300, docker=False):
    logger.setLevel(log_level)
    smt2 = z3_to_dreal(s)
    with open(SMT2_PATH, "w") as smt2file:
        smt2file.write(smt2)
    try:
        if docker: command = DOCKER_COMMAND(delta)
        else: command = [DREAL_PATH, '--precision', str(delta), SMT2_PATH]
        logger.info("Running...")
        result : subprocess.CompletedProcess[bytes] = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        error_msg = result.stderr.decode('utf-8')[:-1]
        if error_msg:
            logger.error("dreal ran into an error:\n%s", error_msg)
            sys.exit(1)
        logger.info("dreal ran successfully.")
        output = result.stdout.decode('utf-8')
        logger.debug("dreal output:\n" + output)
        sat = output[:output.index("\n")]
        sat = DREAL_SAT if "delta-sat" in sat else sat
        model = output[output.index("\n") + 1:-1]
        return sat, model
    except subprocess.TimeoutExpired as e:
        logger.warning("dReal timed out after " + str(timeout) + " seconds.")
        return DREAL_UNKOWN, []
    finally:
        os.remove(SMT2_PATH)

def z3_to_dreal(s : z3.Solver):
    smt2 = s.to_smt2()
    smt2 = smt2[:smt2.index("(")] + ";" + smt2[smt2.index("("):]
    smt2 = smt2[:smt2.index("(check-sat)")]
    smt2 += '(check-sat)\n'
    smt2 += '(get-model)\n'
    smt2 += '(exit)\n'
    return smt2