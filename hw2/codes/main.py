import logging, logging.handlers
import sys
from settings import *
from check_file import check_file, load_shelve
from read_input import read_train_datas, read_train_datas_by_name, read_map
from utils import phi
import re

def init():
    init_log_settings()
    check_file()
    load_shelve()

def main():
    init()
    #dt = read_train_datas(1000)
    #print(dt[0][0])
    #dt = read_train_datas_by_name('faem0_si1392')
    #mp = read_map()
    #res = phi(dt, mp)
    #print(res)

    names = []
    with open(DATA_PATH['test'], 'r') as f:
        for line in f:
            names.append(line.split()[0])

    regex = re.compile('(\w+)_(\d+)')
    with open('b.ans', 'w') as f:
        f.write('id,phone_sequence\n')
        for i in range(len(names)):
            g = regex.match(names[i])
            a, b = g.group(1), int(g.group(2))
            if b == 1:
                f.write('{},\n'.format(a))




if __name__ == '__main__':
    main()
