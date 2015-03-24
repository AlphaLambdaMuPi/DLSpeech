import read_data
from sklearn import linear_model, cross_validation, svm, metrics, grid_search

def predict_submit(model, smpath, outpath, pmpath):
    pmap = []
    for line in open(pmpath):
        x = line.strip('\n').split()
        pmap.append(x[1])

    X_submit, label_submit = read_data.read_feature(smpath, label=True)
    Y_submit = model.predict(X_submit)
    f = open(outpath, 'w')
    f.write('Id,Prediction\n')
    for i in range(len(label_submit)):
        f.write(label_submit[i] + ',' + pmap[Y_submit[i]] + '\n')
    f.close()

def main():
    feature_path = '../data/train_100000.ark'
    label_path = '../data/train_100000.lab'
    submit_feature_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/fbank/test.ark'
    phone_map_path = '../data/phone_map'
    p48_39_path = '../data/48_39.map'

    DATA_SIZE = 10000
    X = read_data.read_feature(feature_path, DATA_SIZE)
    Y = read_data.read_label(label_path, phone_map_path, DATA_SIZE)
    # print(X, Y)
    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
        # X, Y, test_size=0.5)
    
    train_size = len(Y) * 0.5
    train_size = int(train_size)

    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
         ]

    model = svm.SVC(kernel='poly', C=1E-2, gamma=1E-2, degree=2)
    # model = svm.LinearSVC(C=1E0)
    # model = linear_model.LogisticRegression()
    model.fit(X_train, Y_train)
    Y_tpred = model.predict(X_train)
    Y_pred = model.predict(X_test)
    
    # print(Y_test, Y_pred)

    Ein = metrics.zero_one_loss(Y_train, Y_tpred)
    Etest = metrics.zero_one_loss(Y_test, Y_pred)
    print('Ein = {0}'.format(Ein))
    print('Etest = {0}'.format(Etest))

    # predict_submit(model, submit_feature_path, 'test.csv', p48_39_path)
    
    # orig_path = '/home/step5/MLDS_Data/MLDS_HW1_RELEASE_v1/'
    # sort_label(orig_path + 'fbank/train.ark', orig_path + 'state_label/train.lab', 
               # orig_path + 'state_label/train_sorted.lab')

if __name__ == '__main__':
    main()
