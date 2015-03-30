import numpy as np
import read_data
import label_error
from sklearn import linear_model, cross_validation, svm, metrics, grid_search, preprocessing
from theano_test import LogisticRegression
from neural_network import DNN

def predict_submit(model, smpath, outpath, pmpath):
    X_submit, label_submit = read_data.read_feature(smpath, label=True)
    # Y_submit = model.predict(X_submit)
    Y_submit = label_error.transform_label(model.predict(X_submit), pmpath)
    f = open(outpath, 'w')
    f.write('Id,Prediction\n')
    for i in range(len(label_submit)):
        f.write(label_submit[i] + ',' + Y_submit[i] + '\n')
    f.close()

def main():
    orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # feature_path = '../data/train_100000.ark'
    # label_path = '../data/train_100000.lab'
    feature_path = orig_path + 'fbank/train.ark'
    label_path = orig_path + 'label/train_sorted.lab'
    submit_feature_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/fbank/test.ark'
    phone_map_path = '../data/phone_map'
    p48_39_path = '../data/48_39.map'

    DATA_SIZE = 100000
    X = read_data.read_feature(feature_path, DATA_SIZE)
    Y = read_data.read_label(label_path, p48_39_path, DATA_SIZE)
    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
        # X, Y, test_size=0.5)

    # Y = np.sign(Y-19.5)
    
    train_size = len(Y) * 0.9
    train_size = int(train_size)

    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
         ]

    model = DNN([X.shape[1], 200, 200, np.max(Y)+1])
    # model = svm.SVC(kernel='poly', C=1E-2, gamma=1E-2, degree=2)
    # model = svm.LinearSVC(C=1E0)
    # model = linear_model.LogisticRegression()
    model.fit(X_train, Y_train)
    Y_tpred = model.predict(X_train)
    Y_pred = model.predict(X_test)

    # print(Y_test, Y_pred)

    # Ain = 1 - metrics.zero_one_loss(Y_train, Y_tpred)
    # Aval = 1 - metrics.zero_one_loss(Y_test, Y_pred)
    pm = label_error.get_pmap()
    print(metrics.classification_report(Y_train, Y_tpred, target_names=pm))
    Ain = label_error.calc_accuracy(Y_train, Y_tpred)
    Aval = label_error.calc_accuracy(Y_test, Y_pred)
    print('Ain = {0}'.format(Ain))
    print('Aval = {0}'.format(Aval))

    predict_submit(model, submit_feature_path, 'test.csv', p48_39_path)
    
    # orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # sort_label(orig_path + 'fbank/train.ark', orig_path + 'state_label/train.lab', 
               # orig_path + 'state_label/train_sorted.lab')

if __name__ == '__main__':
    main()
