from src.utils import *

import logging
import os
import subprocess
import sys

import z3

DREAL_PATH = '/opt/dreal/4.21.06.2/bin/dreal'

logger = logging.getLogger("dreal")

def run_dreal(s : z3.Solver, delta=0.001):
    smt2 = z3_to_dreal(s)
    smt2_path = '/tmp/barrierforqcirc.smt2'
    with open(smt2_path, "w") as smt2file:
        smt2file.write(smt2)
    try:
        command = [DREAL_PATH, '--precision', str(delta), smt2_path]
        logger.info("Running...")
        result : subprocess.CompletedProcess[bytes] = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    finally:
        os.remove(smt2_path)

def z3_to_dreal(s: z3.Solver):
    smt2 = s.to_smt2()
    smt2 = smt2[:smt2.index("(")] + ";" + smt2[smt2.index("("):]
    smt2 = smt2[:smt2.index("(check-sat)")]
    # smt2 = "(set-logc QF_NRA)\n" + smt2
    smt2 += '(check-sat)\n'
    smt2 += '(get-model)\n'
    smt2 += '(exit)\n'
    return smt2