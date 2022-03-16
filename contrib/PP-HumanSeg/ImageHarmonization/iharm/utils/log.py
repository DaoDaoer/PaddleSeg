import io
import time
import logging
from datetime import datetime

import numpy as np

LOGGER_NAME = 'root'
LOGGER_DATEFMT = '%Y-%m-%d %H:%M:%S'

handler = logging.StreamHandler()

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def add_new_file_output_to_logger(logs_path, prefix, only_message=False):
    log_name = prefix + datetime.strftime(datetime.today(),
                                          '%Y-%m-%d_%H-%M-%S') + '.log'
    logs_path.mkdir(exist_ok=True, parents=True)
    stdout_log_path = logs_path / log_name

    fh = logging.FileHandler(str(stdout_log_path))

    fmt = '%(message)s' if only_message else '(%(levelname)s) %(asctime)s: %(message)s'
    formatter = logging.Formatter(fmt=fmt, datefmt=LOGGER_DATEFMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None, mininterval=5):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if len(self.buf
               ) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = time.time()


class ScalarAccumulator(object):
    def __init__(self, period):
        self.sum = 0
        self.cnt = 0
        self.period = period

    def add(self, value):
        self.sum += value
        self.cnt += 1

    @property
    def value(self):
        if self.cnt > 0:
            return self.sum / self.cnt
        else:
            return 0

    def reset(self):
        self.cnt = 0
        self.sum = 0

    def is_full(self):
        return self.cnt >= self.period

    def __len__(self):
        return self.cnt
