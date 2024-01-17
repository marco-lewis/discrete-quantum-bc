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

# Source: https://stackoverflow.com/a/66209331/19768075
class LoggerWriter:
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self.buffer = []

    def write(self, msg : str):
        if msg.endswith('\n'):
            self.buffer.append(msg.removesuffix('\n'))
            self.logger_func(''.join(self.buffer))
            self.buffer = []
        else:
            self.buffer.append(msg)

    def flush(self):
        pass