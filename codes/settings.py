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
ORIG_PATH = path.abspath('../easy_data/')
FEATURE_PATH = path.join(ORIG_PATH, 'train_100000.ark')
LABEL_PATH = path.join(ORIG_PATH, 'train_100000.lab')
P48_39_PATH = path.join(ORIG_PATH, '48_39.map')

SUBMIT_FEATURE_PATH = '../raw_data/MLDS_HW1_RELEASE_v1/fbank/test.ark'
SUBMIT_FEATURE_PATH_2 = '../easy_data/train_100000.ark'

RESULT_PATH = path.abspath('../result')
# phone_map_path = '../data/phone_map'
