from urllib.request import urlretrieve
import tarfile, os
import logging
from settings import *

logger = logging.getLogger()

def check_file():
    if os.path.isdir(DATA_DIR):
        logger.info('It seems like you have the datas...')
        return

    logger.warning('data/ dir missing, try extract from hw2.tgz...')
    try:
        with tarfile.open(TAR_FILE) as tar:
            data_dir = os.path.dirname(__file__)
            tar.extractall(path=DATA_DIR)
    except:
        logger.error('File {} not found, please download' 
                'from kaggle and extract it here'.format(os.path.abspath(TAR_FILE)))

