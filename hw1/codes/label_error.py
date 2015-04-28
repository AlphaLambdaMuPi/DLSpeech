import numpy as np
from settings1 import *
import sys
sys.path = ['../../hw2/codes'] + sys.path
from hw2 import utils, phone_maps
from hw2.utils import *
from hw2.settings import *
phomap, phomap39, invphomap, labels = [], [], [], []
pho_init()

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

def calc_accuracy2(Y, Yt):
    # return np.average(Y == Yt)
    ans = 0
    cnt = 0
    for y1, y2 in zip(Y, Yt):
        Y1 = answer([id2ph(x) for x in y1], False)
        # Y2 = answer([id2ph(x) for x in clr(y2)], False)
        Y2 = answer([id2ph(x) for x in y2], False)
        print(Y1, ',\n', Y2)
        ans += delta(Y1, Y2)
        cnt += 1
    return ans / cnt

def calc_accuracy3(Y, Yt):
    # return np.average(Y == Yt)
    ans = 0
    cnt = 0
    for y1, y2 in zip(Y, Yt):
        ans += np.sum(y1 == y2)
        cnt += len(y1)
    return ans / cnt

def clr(s):
    ret = []
    lst = ''
    cnt = 0
    for x in s:
        if x != lst:
            if cnt >= 3:
                ret.append(lst)
            lst = x
            cnt = 0
        cnt += 1
    if cnt >= 3:
        ret.append(lst)
    return ret
