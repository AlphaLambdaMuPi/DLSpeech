import numpy as np
import read_data
import label_error
from sklearn import linear_model, cross_validation, svm, metrics, grid_search, preprocessing
from theano_test import LogisticRegression
from neural_network import DNN
import mnist

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
    dims = [X_train.shape[1], 400, 300, np.max(Y_train)+1]
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
    orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # feature_path = '../data/train_100000.ark'
    # label_path = '../data/train_100000.lab'
    feature_path = orig_path + 'fbank/train.ark'
    label_path = orig_path + 'label/train_sorted.lab'
    # submit_feature_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/fbank/test.ark'
    submit_feature_path = '../data/train_100000.ark'
    # phone_map_path = '../data/phone_map'
    p48_39_path = '../data/48_39.map'

    DATA_SIZE = 30000
    X = read_data.read_feature(feature_path, DATA_SIZE)
    Y = read_data.read_label(label_path, p48_39_path, DATA_SIZE)
    # X = X[100000:,:]
    # Y = Y[100000:]

    train_size = len(Y) * 0.5
    train_size = int(train_size)


    perm = np.random.permutation(train_size)
    perm = np.concatenate((perm, list(range(train_size,len(Y)))))
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

    predict_submit(model, submit_feature_path, 'test.csv', p48_39_path)
    
    # orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # sort_label(orig_path + 'fbank/train.ark', orig_path + 'state_label/train.lab', 
               # orig_path + 'state_label/train_sorted.lab')

if __name__ == '__main__':
    main()
