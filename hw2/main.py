import logging, logging.handlers
import sys
from settings import *
from check_file import check_file
from read_input import read_feature_by_groups, read_label_dict, read_map


def main():
    init_log_settings()
    check_file()
    ft = read_feature_by_groups()
    lb = read_label_dict()
    mp = read_map()
    print(mp)
    res = get_psi(ft, lb, mp)
    print(res)

    with open('a.ans', 'w') as f:
        f.write('id,feature\n')
        for i in range(len(res)):
            f.write('faem0_si1392_{},{:.6f}\n'.format(i, res[i]))

def get_psi(ft, lb, mp):
    PHONES = 48
    FBANKS = 69
    DIMS = PHONES * FBANKS + PHONES * PHONES
    res = [0.0] * DIMS
    #res = [0.0] * PHONES * FBANKS + [1000.0] * PHONES * PHONES
    def label_id(l):
        return mp[l][0] 

    def label_feature_id(l, f):
        return l * FBANKS + f
    def trans_id(a, b):
        return PHONES * FBANKS + a * PHONES + b

    name = 'faem0_si1392'
    data = ft[name]
    last_lid = -1
    for i in range(len(data)):
        frame_id = data[i][0]
        lid = label_id(lb[name, frame_id])

        for j in range(FBANKS):
            res[label_feature_id(lid, j)] += data[i][j+1]

        if last_lid != -1:
            res[trans_id(last_lid, lid)] += 1.0

        last_lid = lid

    return res

if __name__ == '__main__':
    main()
