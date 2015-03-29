import numpy as np
import theano.tensor as T
from theano import function, shared, pp
from theano.printing import debugprint

def sigmoid(x):
    return 1 / (1 + T.exp(-x))

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

def LabelToBinary(Y, K):
    N = Y.shape[0]
    lst = [1*(Y==i).reshape(N, 1) for i in range(K)]
    return np.concatenate(lst, axis=1)

class LogisticRegression:
    def __init__(self):
        self._w = None
        self._K = 2

    def fit(self, X, Y, Eta=1E1):
        print(X, Y)
        self._K = int(np.max(Y)) + 1
        N_train = X.shape[0]
        D = X.shape[1]
        x = T.matrix('x')
        y = T.matrix('y')
        # y = yy.dimshuffle(0, 'x')
        w = shared(np.zeros((D, self._K)), name='w')
        eta = shared(Eta, name='eta')

        dotted = T.exp(T.dot(x, w))
        dsm = T.sum(dotted, axis=1).dimshuffle(0, 'x')
        yln = y * T.log(dotted / dsm)

        ein = - (1 / N_train) * T.sum(yln)
        ein_grad = T.grad(ein, w)
        # sm = sigmoid(-y * T.dot(x, w)) * (-y)
        # print(sm.type())
        # ein_grad = T.sum(x + sm) / N_train
        
        ein_func = function([x, y], ein)
        update_func = function([x, y], None, updates=[(w, w - eta * ein_grad), (eta, T.max((eta * 0.9997, 1E-2)))])
        # eg = function([x, y], ein_grad)

        Ein = 1.0
        Batch_size = 100
        Batch_num = N_train // Batch_size
        Epoch = 0

        YY = LabelToBinary(Y, self._K)

        for i in range(2000000 // Batch_size):
            Bno = i % Batch_num
            X_mb = X[Batch_size*Bno:Batch_size*(Bno+1),:]
            Y_mb = YY[Batch_size*Bno:Batch_size*(Bno+1),:]
            update_func(X_mb, Y_mb)
            if Bno == Batch_num - 1:
                Epoch += 1
                Ein = ein_func(X, YY)
                print('Epoch {0}, \tEin = {1}'.format(Epoch, Ein))
            # print(w.get_value())

        self._w = w.get_value()


    def predict(self, X):
        N_test = X.shape[0]

        yp = np.dot(X, self._w)
        return yp.argmax(axis=1)

        return np.zeros((N_test, 1), dtype='int')


