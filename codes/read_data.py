import numpy as np
from sklearn import preprocessing

def preproc(X):
    return X
    return preprocessing.normalize(X)

def read_label(path, pmpath, datasize=1E9):
    pmap = {}
    for i, line in enumerate(list(open(pmpath))):
        x = line.strip('\n').split()
        pmap[x[0]] = i

    arr = []
    data_count = 0

    for line in open(path):
        x = line.strip('\n').split(',')
        arr.append(pmap[x[1]])
        data_count += 1
        if data_count >= datasize:
            break

    print('Label readed : {0}'.format(data_count))
    return np.array(arr)

def read_feature(path, datasize=1E9, label=False):
    arr = []
    lab = []
    data_count = 0
    for line in open(path):
        ln = line.strip('\n').split()
        x = [float(a) for a in ln[1:]]
        arr.append(x)
        if label:
            lab.append(ln[0])
        data_count += 1
        if data_count >= datasize:
            break

    print('Feature readed : {0}'.format(data_count))
    if label:
        return np.array(arr), lab
    return preproc(np.array(arr))

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
    feature_path = '../data/train_10000.ark'
    label_path = '../data/train_10000.lab'
    phone_map_path = '../data/phone_map'

    DATA_SIZE = 10E33
    X = read_feature(feature_path, DATA_SIZE)
    Y = read_label(label_path, phone_map_path, DATA_SIZE)
    # print(X, Y)
    
    # orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # sort_label(orig_path + 'fbank/train.ark', orig_path + 'state_label/train.lab', 
               # orig_path + 'state_label/train_sorted.lab')

if __name__ == '__main__':
    main()

