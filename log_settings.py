import logging

logging.basicConfig(format="(%(relativeCreated)dms)%(name)s:%(levelname)s:%(message)s",datefmt="%H:%M:%S")

def setup_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/" + filename, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("(%(relativeCreated)dms)%(name)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger