import numpy as np

pirho = '../data/48_39.map'

def transform_label(Y, pmpath=pirho):
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

def get_pmap(pmpath=pirho):
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
