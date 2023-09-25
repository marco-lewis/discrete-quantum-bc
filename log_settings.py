import logging

logging.basicConfig(format="(%(relativeCreated)dms)%(name)s:%(levelname)s:%(message)s",datefmt="%H:%M:%S")

def setup_logger(filename, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    fh = logging.FileHandler("logs/" + filename, mode="w")
    fh.setLevel(log_level)
    formatter = logging.Formatter("(%(relativeCreated)dms)%(name)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger