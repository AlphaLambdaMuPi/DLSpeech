import numpy as np
import theano.tensor as T
from theano import function, shared, pp, config
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import debugprint
import time

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

class DNN:
    def __init__(self, dims):
        rng = np.random.RandomState()
        self._Dims = dims
        self._L = len(self._Dims)
        self._x = T.matrix('x', config.floatX)
        self._y = T.matrix('y', config.floatX)
        self._w = []
        self._b = []
        self._l = [self._x]
        for i in range(self._L-1):
            alp = self._Dims[i] ** 0.5
            self._w.append(shared(rng.randn(self._Dims[i], self._Dims[i+1]).astype(config.floatX) / alp))
            self._b.append(shared(rng.randn(self._Dims[i+1]).astype(config.floatX) / alp))
            T.addbroadcast(self._b[i], 0)
        for i in range(self._L-1):
            if i == self._L-2:
                layer = T.dot(self._l[i], self._w[i]) + self._b[i]
            else:
                layer = sigmoid(T.dot(self._l[i], self._w[i]) + self._b[i])
            self._l.append(layer)
        self._h = self._l[-1]

        dotted = T.exp(self._h)
        dsm = T.sum(dotted, axis=1).dimshuffle(0, 'x')
        yln = self._y * T.log(dotted / dsm)

        self._j = - (1 / self._x.shape[0]) * T.sum(yln)
        self._j_grad_w = []
        self._j_grad_b = []
        for i in range(self._L-1):
            self._j_grad_w.append(T.grad(self._j, self._w[i]))
            self._j_grad_b.append(T.grad(self._j, self._b[i]))
        self._eta = shared(np.asarray(0).astype(config.floatX))

    def target_func(self):
        return function([self._x, self._y], self._j)

    def update_func(self, degrade_rate, min_eta):
        udlist = []
        for i in range(self._L-1):
            udlist.append((self._w[i], self._w[i] - self._eta * self._j_grad_w[i]))
            udlist.append((self._b[i], self._b[i] - self._eta * self._j_grad_b[i]))
        udlist.append((self._eta, T.max((self._eta * degrade_rate, min_eta))))
        ret = function([self._x, self._y], None, updates=udlist)
        return ret

    def predict_func(self):
        return function([self._x], self._h)

    def fit(self, X, YY, Eta=3E0):
        X = X.astype(config.floatX)
        N_train = X.shape[0]
        print(X, YY)
        self._K = int(np.max(YY)) + 1
        Y = LabelToBinary(YY, self._K).astype(config.floatX)

        Batch_size = 128
        Batch_num = N_train // Batch_size

        Eta = 6E-2
        degrade_rate = 0.999995 #1 - 5E-2 * (Batch_size / N_train)
        min_eta = Eta * 0.2

        self._eta.set_value(np.asarray(Eta).astype(config.floatX))
        jfunc = self.target_func()
        ufunc = self.update_func(degrade_rate, min_eta)

        Ein = 1.0
        Epoch = 0

        # x_shared = shared(X)
        # y_shared = shared(Y)

        current_j = jfunc(X, Y)
        t0 = time.time()
        for i in range(500 * Batch_num):
            try:
                Bno = i % Batch_num
                b_start = Batch_size * Bno
                b_end = b_start + Batch_size
                X_mb = X[b_start:b_end,:]
                Y_mb = Y[b_start:b_end,:]
                ufunc(X_mb, Y_mb)
                if Bno == Batch_num - 1:
                    Epoch += 1
                    J = jfunc(X, Y)
                    print('Epoch {0}, \tJ = {1:.5f}, \ttime = {2:.3f}, \teta = {3:.5f}'.format(
                        Epoch, float(J), float(time.time()-t0), float(self._eta.get_value()))
                    )
                    t0 = time.time()

                    # new_eta = self._eta.get_value()
                    # if J > current_j:
                        # new_eta *= 0.9
                    # else:
                        # new_eta *= 1.01
                        # if i > Batch_num * 15:
                            # new_eta *= 1.1
                    # self._eta.set_value(new_eta)
                    current_j = J
                    # for i in range(0, 1):
                        # print(self._w[i].get_value())
                        # print(self._b[i].get_value())
            except KeyboardInterrupt:
                break

    def predict(self, X):
        X = X.astype(config.floatX)
        N_test = X.shape[0]

        pfunc = self.predict_func()
        yp = pfunc(X)

        # newm = np.zeros(yp.shape)
        # for i in range(N_test):
            # for j in range(i-2, i+2):
                # if j < 0 or j >= N_test:
                    # continue
                # newm[j] += yp[i]
        # yp = newm

        return yp.argmax(axis=1)

