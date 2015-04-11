import logging, sys, os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_FILE_NAME = os.path.join(ROOT_DIR, 'log/hw2.log')
LOG_LEVEL = logging.DEBUG
TAR_FILE = os.path.join(ROOT_DIR, 'hw2.tgz')

def init_log_settings():
    rotator = logging.handlers.RotatingFileHandler(
                  LOG_FILE_NAME, maxBytes=65536, backupCount=5)
    logging.basicConfig(handlers=[rotator],
                        format='[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s',
                        datefmt='%H:%M:%S',
                        level=LOG_LEVEL)
    logger = logging.getLogger()

    formatter = logging.Formatter(fmt = '[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s'
                                  ,datefmt = '%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

