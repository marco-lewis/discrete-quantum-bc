import logging
from math import floor

class RelativeTimeFormatter(logging.Formatter):
    def format(self, record):
        millisecs = floor(record.relativeCreated % 1000)
        seconds = floor((record.relativeCreated/1000) % 60)
        minutes = floor((record.relativeCreated/(60 * 1000)) % 60)
        hours = floor((record.relativeCreated/(60 * 60 * 1000)) % 60)
        record.relativeTime = str(hours).zfill(1) + ":" + str(minutes).zfill(2) + ":" + str(seconds).zfill(2) + "." + str(millisecs).zfill(3) 
        return super(RelativeTimeFormatter, self).format(record)

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

formatter = RelativeTimeFormatter("(%(relativeTime)s)%(name)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
logging.basicConfig()
logging.root.handlers[0].setFormatter(formatter)

def setup_logger(filename, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    fh = logging.FileHandler("logs/" + filename, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger