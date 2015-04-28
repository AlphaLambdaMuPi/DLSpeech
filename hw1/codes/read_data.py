import numpy as np
import label_error
from sklearn import preprocessing

from settings1 import *

def preproc(X, num = 2):
    mats = []

    for i in range(-num, num+1, 1):
        mats.append(np.concatenate((X[-i*2:, :], X[:-i*2, :]), axis=0))

    Iks = np.concatenate(mats, axis=1)
    Iks = (Iks - np.mean(Iks, axis=0)) / (np.std(Iks, axis=0) + 1E-2)
    # return preprocessing.normalize(X)
    return Iks

def read_label(path, pmpath, datasize=1E9):
    pmap = {}
    with open(pmpath) as f:
        for i, line in enumerate(list(f)):
            x = line.strip('\n').split()
            pmap[x[0]] = i

    arr = []
    data_count = 0

    with open(path) as f:
        for line in f:
            x = line.strip('\n').split(',')
            arr.append(pmap[x[1]])
            data_count += 1
            if data_count >= datasize:
                break

    print('Label readed : {0}'.format(data_count))
    # return label_error.transform_label(np.array(arr))
    return np.array(arr)

def read_group_label(path, pmpath, lab, datasize=1E9):
    Y = read_label(path, pmpath, datasize)
    last_label = ''
    ret = []
    nowy = []
    for f, l in zip(Y, lab):
        pfx = l[:l.rfind('_')]
        if pfx != last_label:
            if last_label != '':
                ret.append(np.array(nowy))
            last_label = pfx
            nowy = []
        nowy.append(f)
    ret.append(np.array(nowy))
    ret = np.array(ret)
    return ret

def read_feature(path, datasize=10**9, label=False):
    arr = []
    lab = []
    data_count = 0
    for line in open(path):
        ln = line.strip('\n').split()
        if ln[0] == 'f':
            alpha = 1.0
        else:
            alpha = -1.0
        x = [alpha] + [float(a) for a in ln[1:]]
        arr.append(x)
        if label:
            lab.append(ln[0])
        data_count += 1
        if data_count >= datasize:
            break

    print('Feature readed : {}'.format(data_count))

    ret = preproc(np.array(arr))
    return (ret, lab) if label else ret

def read_group_feature(path, datasize=10**9, label=False):
    feat, lab = read_feature(path, datasize, True)
    last_label = ''
    lablist = []
    featlist = []
    nowx = []
    for f, l in zip(feat, lab):
        pfx = l[:l.rfind('_')]
        if pfx != last_label:
            if last_label != '':
                featlist.append(np.array(nowx))
            nowx = []
            last_label = pfx
        nowx.append(f)
    featlist.append(np.array(nowx))
    featlist = np.array(featlist)
    print('Group count : {}'.format(len(featlist)))
    return (featlist, lablist, lab) if label else featlist

def sort_label(fpath, lpath, newlpath):
    names = []
    for line in open(fpath):
        names.append(line.split()[0])
    mp = {}
    for line in open(lpath):
        alp = line.strip('\n').split(',')
        mp[alp[0]] = alp[1]
    f = open(newlpath, 'w')
    for nm in names:
        f.write(nm + ',' + mp[nm] + '\n')
    f.close()

def main():
    DATA_SIZE = 10E33
    X = read_feature(FEATURE_PATH, DATA_SIZE)
    Y = read_label(LABEL_PATH, P48_39_PATH, DATA_SIZE)

if __name__ == '__main__':
    main()

