import logging, logging.handlers
import sys
from settings import *
from check_file import check_file
from read_input import read_feature_by_groups, read_label_dict


def main():
    init_log_settings()
    check_file()
    ft = read_feature_by_groups()
    lb = read_label_dict()
    mp = read_map()
    res = get_psi()
    print(res)

def get_psi(ft, lb, mp):
    PHONES = 48
    FBANKS = 69
    DIMS = PHONES * FBANKS + PHONES * PHONES
    res = [0.0] * DIMS

    def label_id(l):
        return mp[l][0]

    def label_feature_id(l, f):
        return label_id(l) * FBANKS + f
    def trans_id(a, b):
        return PHONES * FBANKS + label_id(a) * PHONES + label_id(b)

    name = 'faem0_sil1392'
    data = ft[name]
    for i in range(data):
        frame_id = data[i][0]
        lid = label_id(lb[name, frame_id])
        for j in range(FBANKS):
            res[label_feature_id(lid, j)] += data[j+1]

        if i > 0:
            res[trans_id(lid, label_id(lb[name, frame_id]))] += 1.0

    return res
        

if __name__ == '__main__':
    main()
