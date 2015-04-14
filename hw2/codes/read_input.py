from settings import *
import logging
import re
logger = logging.getLogger()

def read_feature(max_lines = 10**3):

    cnt = 0
    feature = []
    with open(DATA_PATH['fbank']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split()
            datas = datas[0:1] + list(map(float, datas[1:]))
            feature.append(datas)
            cnt += 1
            if cnt >= max_lines:
                break

    logger.info('Read %d datas feature...', cnt)

    return feature

def read_feature_by_groups():

    fe = read_feature()
    nid = 0
    res = {}
    while nid < len(fe):
        gr = []
        cur_name, __ = feature_group(fe[nid][0])
        i = nid
        while i < len(fe) and cur_name == feature_group(fe[i][0])[0]:
            ls = [feature_group(fe[i][0])[1]] + fe[i][1:]
            gr.append(ls)
            i += 1
        res[cur_name] = gr
        nid = i
    return res

regex = re.compile(r'(\w+)_(\d+)')
def feature_group(name):
    
    match = regex.fullmatch(name)
    return match.group(1), int(match.group(2))

def read_label(max_lines = 400000):

    cnt = 0
    res = []
    with open(DATA_PATH['label']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split(',')
            res.append(datas)
            if len(datas) != 2:
                raise ValueError('zz')
            cnt += 1
            if cnt >= max_lines:
                break

    logger.info('Read %d datas label...', cnt)
    return res

def read_label_dict():
    ls = read_label()
    res = {}
    for data in ls:
        pr = feature_group(data[0])
        res[pr] = data[1]

    return res
    
def read_map():
    res = {}
    with open(DATA_PATH['48_idx_chr']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split()
            print(datas[0])
            res[datas[0]] = int(datas[1]), datas[2]

    return res


