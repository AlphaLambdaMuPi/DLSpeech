import numpy as np
import read_data
import label_error
from sklearn import linear_model, cross_validation, svm, metrics, grid_search, preprocessing
from theano_test import LogisticRegression
from neural_network import DNN
import mnist

import os
import datetime as dt

from settings import *

def predict_submit(model, smpath, outpath, pmpath):
    X_submit, label_submit = read_data.read_feature(smpath, label=True)
    # Y_submit = model.predict(X_submit)
    Y_submit = label_error.transform_label(model.predict(X_submit), pmpath)
    f = open(outpath, 'w')
    f.write('Id,Prediction\n')
    for i in range(len(label_submit)):
        f.write(label_submit[i] + ',' + Y_submit[i] + '\n')
    f.close()

def train_experiment(X_train, Y_train, X_test, Y_test, epoch=20):
    # model = svm.SVC(kernel='poly', C=1E-2, gamma=1E-2, degree=2)
    # model = svm.LinearSVC(C=1E0)
    # model = linear_model.LogisticRegression()
    dims = [X_train.shape[1], 100, 100, 100, np.max(Y_train)+1]
    param_grid = [
          {
              'Dims': [dims],
              'Eta': [3E-3, 1E-2, 3E-2, 1E-1], 
              'Drate': [0.9999, 0.99999, 0.99999],
              'Minrate': [0.2], 
              'Momentum': [0.0, 0.5, 0.9],
              'Batchsize': [128],
          },
    ]

    # param_grid = [
          # {
              # 'Dims': [dims],
              # 'Eta': [3E-3, 1E-2], 
              # 'Drate': [0.9999],
              # 'Minrate': [0.2], 
              # 'Momentum': [0.0],
              # 'Batchsize': [128],
          # },
    # ]

    # model = DNN(dims)
    # clf = grid_search.GridSearchCV(model, param_grid, score_func=metrics.accuracy_score, verbose=True)
    # clf.fit(X_train, Y_train)

    # print(clf.best_params_)

    model = DNN(
        dims,
        Eta = 0.002, Drate = 0.99998, Minrate = 0.2, Momentum = 0.9, 
        Batchsize = 128
    )

    model.fit(X_train, Y_train, X_test, Y_test, N_epoch=epoch)
    Y_tpred = model.predict(X_train)
    Y_pred = model.predict(X_test)

    # print(Y_test, Y_pred)

    # Ain = 1 - metrics.zero_one_loss(Y_train, Y_tpred)
    # Aval = 1 - metrics.zero_one_loss(Y_test, Y_pred)
    pm = label_error.get_pmap()
    # print(metrics.classification_report(Y_train, Y_tpred, target_names=pm))
    Ain = label_error.calc_accuracy(Y_train, Y_tpred)
    Aval = label_error.calc_accuracy(Y_test, Y_pred)
    print('Ain = {0}'.format(Ain))
    print('Aval = {0}'.format(Aval))

    return Aval, model

def main():

    DATA_SIZE = 300000
    X = read_data.read_feature(FEATURE_PATH, DATA_SIZE)
    Y = read_data.read_label(LABEL_PATH, P48_39_PATH, DATA_SIZE)
    print(type(X))
    X = X[:,:]
    Y = Y[:]
    
    cur_time_string = dt.datetime.now().strftime('%m%d_%H%M%S')
    SUBMIT_PATH = os.path.join(RESULT_PATH, cur_time_string) 
    os.makedirs(SUBMIT_PATH)

    train_size = len(Y) * 0.5
    train_size = int(train_size)


    perm = np.random.permutation(train_size)
    print(perm)
    perm = np.concatenate((perm, list(range(train_size,len(Y)))))
    print(perm)
    X = X[perm,:]
    Y = Y[perm]

    print(X.shape, Y.shape)


    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    # Alpha, Beta, Gamma = mnist.load_data('mnist3.pkl.gz')
    # X_train, Y_train = Alpha
    # X_test, Y_test = Gamma

    Aval, model = train_experiment(X_train, Y_train, X_test, Y_test, 2000)

    predict_submit(model, SUBMIT_FEATURE_PATH, os.path.join(SUBMIT_PATH, 'submit.csv'), P48_39_PATH)
    predict_submit(model, SUBMIT_FEATURE_PATH_2, os.path.join(SUBMIT_PATH, 'test.csv'), P48_39_PATH)
    
    # orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # sort_label(orig_path + 'fbank/train.ark', orig_path + 'state_label/train.lab', 
               # orig_path + 'state_label/train_sorted.lab')

if __name__ == '__main__':
    main()
