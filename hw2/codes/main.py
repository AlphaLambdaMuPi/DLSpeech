import logging, logging.handlers
import sys
from settings import *
from check_file import check_file, load_shelve
from read_input import read_train_datas, read_train_datas_by_name, read_map
from utils import psi

def main():
    init_log_settings()
    check_file()
    load_shelve()
    #dt = read_train_datas(1000)
    #print(dt[0][0])
    dt = read_train_datas_by_name('faem0_si1392')
    mp = read_map()
    res = psi(dt, mp)
    print(res)

    with open('b.ans', 'w') as f:
       f.write('id,feature\n')
       for i in range(len(res)):
           f.write('faem0_si1392_{},{:.6f}\n'.format(i, res[i]))


if __name__ == '__main__':
    main()
