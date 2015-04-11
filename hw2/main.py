import logging, logging.handlers
import sys
import B
from settings import *
from check_file import check_file



def main():
    init_log_settings()
    check_file()


if __name__ == '__main__':
    main()
