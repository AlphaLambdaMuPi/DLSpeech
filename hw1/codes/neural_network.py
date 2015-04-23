import numpy as np
import theano.tensor as T
from theano import function, shared, pp, config
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import debugprint
import label_error
import time
import matplotlib.pyplot as plt
plt.ion()

def sigmoid(x):
    # return 1 / (1 + T.exp(-x))
    # return x / (1 + T.abs_(x))
    return T.maximum(x, 0)
    # return T.maximum(x, 0) + 0.05 * T.minimum(x, 0)
    # return T.log(1+T.exp(x))
    # return T.tanh(x)

# def prelu(x, a):
    # return T.maximum(x, 0) + a * T.minimum(x, 0)

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

def LabelToBinary(Y, K):
    N = Y.shape[0]
    lst = [1*(Y==i).reshape(N, 1) for i in range(K)]
    return np.concatenate(lst, axis=1)

class DNN:
    def get_params(self, **kwargs):
        params = {
            'Dims': self._Dims,
            'Eta': self._Eta,
            'Drate': self._Drate, 
            'Minrate': self._Minrate, 
            'Momentum': self._Momentum, 
            'Batchsize': self._Batchsize,
        }
        return params

    def set_params(self, **kwargs):
        print(kwargs)
        self._Dims = kwargs['Dims']
        self._Eta = kwargs['Eta']
        self._Drate = kwargs['Drate']
        self._Minrate = kwargs['Minrate']
        self._Momentum = kwargs['Momentum']
        self._Batchsize = kwargs['Batchsize']

        return self

    def __init__(self, Dims = [1, 1], 
                 Eta = 0.03, Drate = 0.999998, Minrate = 0.2, Momentum = 0.9,
                 Batchsize = 128):
        self._Dims = Dims
        self._Eta = Eta
        self._Drate = Drate
        self._Minrate = Minrate
        self._Momentum = Momentum
        self._Batchsize = Batchsize

        rng = np.random.RandomState()
        srng = RandomStreams()
        self._L = len(self._Dims)
        self._x = T.matrix('x', config.floatX)
        self._y = T.matrix('y', config.floatX)
        self._prob = 0.5
        self._dropout = False
        self._p = shared(np.array(self._prob).astype(config.floatX))
        self._w = []
        self._b = []
        # self._prelu = []
        # self._penal = 0
        self._r = []
        self._w_delta = []
        self._b_delta = []
        # self._prelu_delta = []
        self._state = []
        # self._penal = 0
        self._l = [self._x]
        for i in range(self._L-1):
            alp = self._Dims[i] ** 0.5
            self._w.append(shared(rng.randn(self._Dims[i], self._Dims[i+1]).astype(config.floatX) / alp))
            self._b.append(shared(rng.randn(self._Dims[i+1]).astype(config.floatX) / alp))
            # self._prelu.append(shared((0.05 * np.ones(self._Dims[i+1])).astype(config.floatX)))
            self._r.append(srng.binomial((self._Dims[i], self._Dims[i+1]), p=self._p, dtype=config.floatX))
            # self._penal += T.mean(self._w[i]**2)
            self._w_delta.append(shared(np.zeros((self._Dims[i], self._Dims[i+1])).astype(config.floatX)))
            self._b_delta.append(shared(np.zeros((self._Dims[i+1])).astype(config.floatX)))
            # self._prelu_delta.append(shared(np.zeros((self._Dims[i+1])).astype(config.floatX)))
        for i in range(self._L):
            self._state.append(shared(np.zeros((self._Batchsize, self._Dims[i])).astype(config.floatX)))

        #recurrent
        self._w.append(shared(rng.randn(self._Dims[-1], self._Dims[-2]).astype(config.floatX) / (self._Dims[-1] ** 0.5 * 2)))
        self._w_delta.append(shared(np.zeros((self._Dims[-1], self._Dims[-2])).astype(config.floatX)))

        for i in range(self._L-1):
            newl = self._l[i]
            neww = self._w[i]
            if self._dropout:
                if i != 0:
                    # #newl = (self._l[i] * self._r[i]) # / (1 - self._p)
                    neww = self._w[i] * self._r[i] #* self._prob / self._p
            if i == self._L-2:
                layer = T.dot(newl, neww) + self._b[i]
            # elif i == 1: # Maxout
                # layer = T.max(T.reshape(newl, (newl.shape[0], self._Dims[i+1], self._Dims[i]//self._Dims[i+1]), ndim=3), axis=2)
            elif i == self._L-3: #recurrent
                layer = sigmoid(T.dot(newl, neww) + T.dot(self._state[i+2], self._w[self._L-1]) + self._b[i])
                # layer = prelu(T.dot(newl, neww) + T.dot(self._state[i+2], self._w[self._L-1]) + self._b[i], self._prelu[i])
            else:
                layer = sigmoid(T.dot(newl, neww) + self._b[i])
                # layer = prelu(T.dot(newl, neww) + self._b[i], self._prelu[i])
            self._l.append(layer)
        self._h = self._l[-1]

        dotted = T.exp(self._h)
        dsm = T.sum(dotted, axis=1).dimshuffle(0, 'x')
        yln = self._y * T.log(dotted / dsm)

        self._j = - (1 / self._x.shape[0]) * T.sum(yln) # + 5 * self._penal
        self._j_grad_w = []
        self._j_grad_b = []
        # self._j_grad_prelu = []
        for i in range(self._L-1):
            self._j_grad_w.append(T.grad(self._j, self._w[i], disconnected_inputs='ignore'))
            self._j_grad_b.append(T.grad(self._j, self._b[i], disconnected_inputs='ignore'))
            # self._j_grad_prelu.append(T.grad(self._j, self._prelu[i], disconnected_inputs='ignore'))
        #recurrent
        self._j_grad_w.append(T.grad(self._j, self._w[self._L-1], disconnected_inputs='ignore'))
        self._eta = shared(np.asarray(0).astype(config.floatX))


    def plt_init(self):
        return
        self.figure, self.ax = plt.subplots()
        self.line_ain, = self.ax.plot([], [], '-o')
        self.line_aval, = self.ax.plot([], [], '-o')
        self.ax.set_autoscaley_on(True)
        self.ax.grid()

    def plt_refresh(self, x, ain, aval):
        return
        self.line_ain.set_xdata(np.append(self.line_ain.get_xdata(), x))
        self.line_ain.set_ydata(np.append(self.line_ain.get_ydata(), ain))
        self.line_aval.set_xdata(np.append(self.line_aval.get_xdata(), x))
        self.line_aval.set_ydata(np.append(self.line_aval.get_ydata(), aval))
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def target_value(self, X, YY):
        # self._p.set_value(1)
        jfunc = function([self._x, self._y], self._j)

        N_test = X.shape[0]

        # TODO Batch_size = 10000
        Batch_size = self._Batchsize
        Batch_num = (N_test + Batch_size - 1) // Batch_size

        tot_j = 0
        for i in range(Batch_num):
            start = Batch_size * i
            end = start + Batch_size
            Xt = X[start:end,:]
            # Yt = Y[start:end]
            Yt = LabelToBinary(YY[start:end], self._K).astype(config.floatX)
            tot_j += jfunc(Xt, Yt) * Xt.shape[0]

        tot_j /= N_test
        return tot_j

    def update_func(self, degrade_rate, min_eta, momentum):
        udlist = []
        for i in range(self._L-2, -1, -1):
            udlist.append((self._w_delta[i], self._w_delta[i] * momentum - self._eta * self._j_grad_w[i]))
            udlist.append((self._b_delta[i], self._b_delta[i] * momentum - self._eta * self._j_grad_b[i]))
            # udlist.append((self._prelu_delta[i], self._prelu_delta[i] * momentum - self._eta * self._j_grad_prelu[i]))
        #recurrent
        udlist.append((self._w_delta[self._L-1], self._w_delta[self._L-1] * momentum - self._eta * self._j_grad_w[self._L-1]))
        for i in range(self._L): #recurrent
            udlist.append((self._state[i], self._l[i]))
        udlist.append((self._eta, T.max((self._eta * degrade_rate, min_eta))))
        ret = function([self._x, self._y], None, updates=udlist)
        return ret

    def realupdate_func(self):
        udlist = []
        for i in range(self._L-2, -1, -1):
            udlist.append((self._w[i], self._w[i] + self._w_delta[i]))
            udlist.append((self._b[i], self._b[i] + self._b_delta[i]))
            # udlist.append((self._prelu[i], self._prelu[i] + self._prelu_delta[i]))
        udlist.append((self._w[self._L-1], self._w[self._L-1] + self._w_delta[self._L-1]))
        ret = function([], None, updates=udlist)
        return ret

    def predict_func(self):
        udlist = []
        for i in range(self._L): #recurrent
            udlist.append((self._state[i], self._l[i]))
        return function([self._x], self._h, updates=udlist)

    def fit(self, X, YY):
        self.X = X.astype(config.floatX)
        # self.X_t = X_t.astype(config.floatX)
        self.YY = YY
        # self.YY_t = YY_t
        self.N_train = self.X.shape[0]
        # print(X, YY)
        self._K = int(np.max(YY)) + 1
        # self.Y_t = LabelToBinary(YY_t, self._K).astype(config.floatX)


        # Eta = 5E-2
        # degrade_rate = 0.999995 #1 - 5E-2 * (Batch_size / N_train)
        # min_eta = Eta * 0.2

        Eta = self._Eta
        degrade_rate = self._Drate
        min_eta = self._Eta * self._Minrate
        momentum = self._Momentum

        self._eta.set_value(np.asarray(Eta).astype(config.floatX))
        self.ufunc = self.update_func(degrade_rate, min_eta, momentum)
        self.rfunc = self.realupdate_func()

        self.Epoch = 0

        # x_shared = shared(X)
        # y_shared = shared(Y)
        return self

    def run_train(self, N_epoch=1):
        t0 = time.time()
        to_break = False

        Batch_size = self._Batchsize
        Batch_num = self.N_train // Batch_size
        for i in range(N_epoch * Batch_num):
            try:
                Bno = i % Batch_num
                b_start = Batch_size * Bno
                b_end = b_start + Batch_size
                # X_mb = self.X[b_start:b_end,:]
                # Y_mb = LabelToBinary(self.YY[b_start:b_end], self._K).astype(config.floatX)
                X_mb = self.X[Bno::Batch_num,:]
                Y_mb = LabelToBinary(self.YY[Bno::Batch_num], self._K).astype(config.floatX)
                # self._p.set_value(self._prob)
                self.ufunc(X_mb, Y_mb)
                self.rfunc()
                if Bno == Batch_num - 1:
                    self.Epoch += 1
                    J = self.target_value(self.X, self.YY)
                    print('Epoch {0}, \tJ = {1:.5f}, \ttime = {2:.3f}, \teta = {3:.5f}'.format(
                        self.Epoch, float(J), float(time.time()-t0), float(self._eta.get_value()))
                    )
                    t0 = time.time()

                    # new_eta = self._eta.get_value()
                    # if J > current_j:
                        # new_eta *= 0.7
                    # else:
                        # new_eta *= 1.1
                        # if i > Batch_num * 15:
                            # new_eta *= 1.1
                    # self._eta.set_value(new_eta)

                    current_j = J

                    # for i in range(0, self._L-1):
                        # ww = self._w[i].get_value()
                        # bb = self._b[i].get_value()
                        # print(i, ww.mean(), ww.std(), bb.mean(), bb.std())
                    if to_break:
                        return False
                    
            except KeyboardInterrupt:
                to_break = True
                # break

        # print(self._w[self._L-1].get_value())
        # for p in self._prelu:
            # print(p.get_value().mean())
        # print('Ain =', Ain)
        return True

    def predict(self, X, prob=False):
        # self._p.set_value(1)
        X = X.astype(config.floatX)
        N_test = X.shape[0]

        # TODO Batch_size = 10000
        Batch_size = self._Batchsize
        Batch_num = (N_test + Batch_size - 1) // Batch_size

        RTIME = 1
        if self._dropout:
            RTIME = 5
        pfunc = self.predict_func()
        ynums = []
        for i in range(RTIME):
            yps = []
            for i in range(Batch_num):
                start = Batch_size * i
                end = start + Batch_size
                # yps.append(pfunc(X[start:end,:]))
                pf = pfunc(X[i::Batch_num,:])
                yps.append(pf.reshape((pf.shape[0], 1, pf.shape[1])))

            yp = np.concatenate(yps, axis=1).reshape((X.shape[0], yps[0].shape[2]))
            ynums.append(yp)
        yp = sum(ynums) / len(ynums)

        # newm = np.zeros(yp.shape)
        # for i in range(N_test):
            # for j in range(i-2, i+2):
                # if j < 0 or j >= N_test:
                    # continue
                # newm[j] += yp[i]
        # yp = newm

        if prob:
            return yp
        return yp.argmax(axis=1)

