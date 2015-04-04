import numpy as np
from settings import *

def transform_label(Y, pmpath=P48_39_PATH):
    pmap = []
    f = open(pmpath)
    for line in f:
        x = line.strip('\n').split()
        pmap.append(x[-1])
    f.close()
    Ys = []
    for i in Y:
        Ys.append(pmap[i])
    return np.array(Ys)

def get_pmap(pmpath=P48_39_PATH):
    pmap = []
    f = open(pmpath)
    for line in f:
        x = line.strip('\n').split()
        pmap.append(x[-1])
    f.close()
    return pmap

def calc_accuracy(Y, Yt):
    # return np.average(Y == Yt)
    Y1 = transform_label(Y)
    Y2 = transform_label(Yt)
    return np.average(Y1 == Y2)
