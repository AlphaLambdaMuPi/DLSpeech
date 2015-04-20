import numpy as np
import read_data
import label_error
from sklearn import linear_model, cross_validation, svm, metrics, grid_search, preprocessing
from theano_test import LogisticRegression
from neural_network import DNN
import mnist

import os
import datetime as dt
from datetime import datetime
import requests

from settings import *

NOW_STR = ''

def predict_submit(models, smpath, outpath, pmpath):
    X_submit, label_submit = read_data.read_feature(smpath, label=True)
    # Y_submit = model.predict(X_submit)
    Y_submit = label_error.transform_label(group_predict(models, X_submit), pmpath)
    f = open(outpath, 'w')
    f.write('Id,Prediction\n')
    for i in range(len(label_submit)):
        f.write(label_submit[i] + ',' + Y_submit[i] + '\n')
    f.close()
    del X_submit, label_submit, Y_submit

def predict_prob(models, smpath, outpath, pmpath):
    X_submit, label_submit = read_data.read_feature(smpath, label=True)
    # Y_submit = model.predict(X_submit)
    Y_submit = group_predict(models, X_submit, prob=True)
    f = open(outpath, 'w')
    for i in range(Y_submit.shape[0]):
        f.write(label_submit[i] + ' ' + ' '.join(str(j) for j in Y_submit[i,:]) + '\n')
    f.close()
    del X_submit, label_submit, Y_submit

# def predict_prob2(model, X_submit, outpath):
    # Y_submit = model.predict(X_submit, prob=True)
    # f = open(outpath, 'w')
    # for i in range(Y_submit.shape[0]):
        # f.write(' '.join(str(j) for j in Y_submit[i,:]) + '\n')
    # f.close()

def group_predict(models, X, prob=False):
    yp = sum([m.predict(X, prob=True) for m in models]) / len(models)
    if prob:
        return yp
    return yp.argmax(axis=1)

def predict_prob(model, smpath, outpath, pmpath):
    X_submit, label_submit = read_data.read_feature(smpath, label=True)
    # Y_submit = model.predict(X_submit)
    Y_submit = model.predict(X_submit, prob=True)
    f = open(outpath, 'w')
    for i in range(Y_submit.shape[0]):
        f.write(' '.join(str(j) for j in Y_submit[i,:]) + '\n')
    f.close()

def predict_prob2(model, X_submit, outpath):
    Y_submit = model.predict(X_submit, prob=True)
    f = open(outpath, 'w')
    for i in range(Y_submit.shape[0]):
        f.write(' '.join(str(j) for j in Y_submit[i,:]) + '\n')
    f.close()

def train_experiment(X_train, Y_train, X_test, Y_test, epoch=20):
    # model = svm.SVC(kernel='poly', C=1E-2, gamma=1E-2, degree=2)
    # model = svm.LinearSVC(C=1E0)
    # print(np.std(X_train, axis=0))
    # print(np.mean(X_train, axis=0))
    # model = linear_model.LogisticRegression()
    dims = [X_train.shape[1], 300, 300, 300, np.max(Y_train)+1]
    # param_grid = [
          # {
              # 'Dims': [dims],

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

    MSIZE = 1
    models = []
    for i in range(MSIZE):
        x = DNN(
            dims,
            Eta = 0.002, Drate = 0.99998, Minrate = 0.2, Momentum = 0.9, 
            Batchsize = 100
        )
        L = X_train.shape[0]
        Rate = 0.8
        if MSIZE == 1:
            Rate = 1
        TL = int(L * Rate)
        perm = np.random.permutation(L)[:TL]
        # x.fit(X_train[perm], Y_train[perm])
        x.fit(X_train, Y_train)
        models.append(x)

    # epoch = 30
    
    models[0].plt_init()
    for i in range(epoch):
        to_break = False
        for model in models:
            if not model.run_train():
                to_break = True
        yp = group_predict(models, X_train)
        Ain = label_error.calc_accuracy(Y_train, yp)
        yp_t = group_predict(models, X_test)
        Aval = label_error.calc_accuracy(Y_test, yp_t)

        print('Ain = {:.4f}, \tAval = {:.4f}'.format(Ain, Aval))
        try:
            requests.post('http://140.112.18.227:5000/post/send_data', 
                         {'name': NOW_STR, 
                          'item': 'Ain',
                          'value': Ain,
                          'time_stamp': str(datetime.now())})
            requests.post('http://140.112.18.227:5000/post/send_data', 
                         {'name': NOW_STR, 
                          'item': 'Aval',
                          'value': Aval,
                          'time_stamp': str(datetime.now())})
        except Exception:
            pass
        models[0].plt_refresh(models[0].Epoch, Ain, Aval)
        if to_break:
            break

    Y_tpred = group_predict(models, X_train)
    Y_pred = group_predict(models, X_test)

    # print(Y_test, Y_pred)

    # Ain = 1 - metrics.zero_one_loss(Y_train, Y_tpred)
    # Aval = 1 - metrics.zero_one_loss(Y_test, Y_pred)
    pm = label_error.get_pmap()
    # print(metrics.classification_report(Y_train, Y_tpred, target_names=pm))
    Ain = label_error.calc_accuracy(Y_train, Y_tpred)
    Aval = label_error.calc_accuracy(Y_test, Y_pred)
    print('Ain = {0}'.format(Ain))
    print('Aval = {0}'.format(Aval))

    return Aval, models

def main():
    DATA_SIZE = 100000
    X = read_data.read_feature(FEATURE_PATH, DATA_SIZE)
    Y = read_data.read_label(LABEL_PATH, P48_39_PATH, DATA_SIZE)
    print(type(X))
    X = X[:,:]
    Y = Y[:]

    global NOW_STR
    cur_time_string = dt.datetime.now().strftime('%m%d_%H%M%S')
    NOW_STR = cur_time_string + '_dnn'
    SUBMIT_PATH = os.path.join(RESULT_PATH, cur_time_string) 
    os.makedirs(SUBMIT_PATH)

    train_size = len(Y) * 0.9
    train_size = int(train_size)

    perm = np.random.permutation(train_size)
    # print(perm)
    perm = np.concatenate((perm, list(range(train_size,len(Y)))))
    print(perm)
    X = X[perm,:]
    Y = Y[perm]

    print(X.shape, Y.shape)


    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    del X, Y

    # Alpha, Beta, Gamma = mnist.load_data('mnist3.pkl.gz')
    # X_train, Y_train = Alpha
    # X_test, Y_test = Gamma
    try:
        requests.post('http://140.112.18.227:5000/post/send_status', 
                     {'name': NOW_STR, 
                      'status': 'new',
                      'type': 'dnn',
                      'time_stamp': str(datetime.now())})
    except Exception:
        pass

    Aval, models = train_experiment(X_train, Y_train, X_test, Y_test, 2000)

    del X_train, X_test, Y_train, Y_test, perm
    
    try:
        requests.post('http://140.112.18.227:5000/post/send_status', 
                     {'name': NOW_STR, 
                      'status': 'ended',
                      })
    except Exception:
        pass


    # predict_submit(models, SUBMIT_FEATURE_PATH_2, os.path.join(SUBMIT_PATH, 'test.csv'), P48_39_PATH)
    # predict_submit(models, SUBMIT_FEATURE_PATH_2, os.path.join(SUBMIT_PATH, 'feature.csv'), P48_39_PATH)
    # predict_prob(models, SUBMIT_FEATURE_PATH_2, os.path.join(SUBMIT_PATH, 'prob.csv'), P48_39_PATH)

    # predict_submit(models, SUBMIT_FEATURE_PATH, os.path.join(SUBMIT_PATH, 'submit.csv'), P48_39_PATH)
    # predict_prob(models, SUBMIT_FEATURE_PATH, os.path.join(SUBMIT_PATH, 'submit_prob.csv'), P48_39_PATH)
    # predict_prob(models, FEATURE_PATH, os.path.join(SUBMIT_PATH, 'prob.csv'), P48_39_PATH)
    
    # orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # sort_label(orig_path + 'fbank/train.ark', orig_path + 'state_label/train.lab', 
               # orig_path + 'state_label/train_sorted.lab')

if __name__ == '__main__':
    main()

