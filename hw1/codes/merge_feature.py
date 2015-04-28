from settings1 import *
from read_data import read_feature

def main():
    FPATH = path.join(ORIG_PATH, 'fbank', 'test.ark')
    MPATH = path.join(ORIG_PATH, 'mfcc', 'test.ark')
    OPATH = path.join(ORIG_PATH, 'mixed', 'test.ark')
    ff = open(FPATH)
    fm = open(MPATH)
    fo = open(OPATH, 'w')

    cnt = 0
    for lf, lm in zip(ff, fm):
        s = lf.strip('\n') + lm[lm.find(' '):]
        fo.write(s)
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)

    ff.close()
    fm.close()
    fo.close()

if __name__ == '__main__':
    main()
