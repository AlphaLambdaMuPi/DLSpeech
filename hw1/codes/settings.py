from os import path

'''
Real Test Data
'''
#ORIG_PATH = path.abspath('../raw_data/MLDS_HW1_RELEASE_v1/')
#FEATURE_PATH = path.join(ORIG_PATH, 'fbank', 'train.ark')
#LABEL_PATH = path.join(ORIG_PATH, 'label', 'train_sorted.lab')
#P48_39_PATH = path.join(ORIG_PATH, 'phones', '48_39.map')

'''
Easy Test Data
'''
ORIG_PATH = path.abspath('../../hw2/data')
FEATURE_PATH = path.join(ORIG_PATH, 'mfcc', 'train.ark')
LABEL_PATH = path.join(ORIG_PATH, 'label', 'train.lab2')
P48_39_PATH = path.join(ORIG_PATH, 'phones', '48_39.map')

SUBMIT_FEATURE_PATH = path.join(ORIG_PATH, 'fbank', 'test.ark')
SUBMIT_FEATURE_PATH_2 = '../easy_data/train_100000.ark'

RESULT_PATH = path.abspath('../result')
# phone_map_path = '../data/phone_map'
