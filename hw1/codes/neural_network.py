import numpy as np
import theano.tensor as T
from theano import function, shared, pp, config, scan
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import debugprint
import label_error
import time
import matplotlib.pyplot as plt
plt.ion()

def sigmoid(x):
    return 1 / (1 + T.exp(-x))

def relu(x):
    return T.maximum(x, 0)

def prelu(x, a):
    return T.maximum(x, 0) + a * T.minimum(x, 0)

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
            'K': self._K,
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
        self._K = kwargs['K']

        return self

    def new_parameters(self, dims):
        w = []
        w_delta = []
        for dim, std in dims:
            if type(dim) == tuple:
                ww = shared(self.rng.randn(dim[0], dim[1]).astype(config.floatX) * std)
            else:
                ww = shared(self.rng.randn(dim).astype(config.floatX) * std)
            ww_delta = shared(np.zeros(dim).astype(config.floatX))
            w.append(ww)
            w_delta.append(ww_delta)
        return w, w_delta


    def feedforward_layer(self, prev_layer, prev_dim, dim):
        alpha = (prev_dim + dim) ** 0.5
        wlh = shared(self.rng.randn(prev_dim, dim).astype(config.floatX) / alpha)
        wlh_delta = shared(np.zeros((prev_dim, dim)).astype(config.floatX))
        b = shared(self.rng.randn(dim).astype(config.floatX) / alpha)
        b_delta = shared(np.zeros(dim).astype(config.floatX))

        layer = relu(T.dot(prev_layer, wlh) + b)
        return layer, [wlh, b], [wlh_delta, b_delta]

    def recurrent_layer(self, prev_layer, prev_dim, dim, backward=False):
        alpha = (prev_dim + dim) ** 0.5
        wlist, wdlist = self.new_parameters([
            (dim, 1),
            ((prev_dim, dim), 1 / alpha),
            ((dim, dim), 1 / alpha),
            (dim, 1 / alpha),
        ])
        h0, wlh, whh, b = wlist
        tmp = h0.dimshuffle('x', 0) + T.zeros((prev_layer.shape[1], dim))

        def func(x, lp):
            return relu(T.dot(x, wlh) + T.dot(lp, whh) + b)
        layer, _ = scan(fn=func, 
                        sequences=prev_layer, 
                        outputs_info=tmp,
                        go_backwards=backward)
        return layer, wlist, wdlist
    
    def bidirectional_recurrent_layer(self, prev_layer, prev_dim, dim):
        pdim = prev_layer.shape[2]
        l1, w1, wd1 = self.recurrent_layer(prev_layer, prev_dim, dim // 2)
        l2, w2, wd2 = self.recurrent_layer(prev_layer, prev_dim, dim // 2, backward=True)
        layer = T.concatenate((l1, l2), axis=2)
        return layer, w1+w2, wd1+wd2

    def lstm_layer(self, prev_layer, prev_dim, dim):
        alpha = (prev_dim + dim) ** 0.5
        ia = 1 / alpha
        wlist, wdlist = self.new_parameters([
            (dim, 1),
            (dim, 1),
            ((prev_dim, dim), ia),
            ((dim, dim), ia),
            (dim, ia),
            ((prev_dim, dim), ia),
            ((dim, dim), ia),
            (dim, ia),
            ((prev_dim, dim), ia),
            ((dim, dim), ia),
            ((prev_dim, dim), ia),
            ((dim, dim), ia),
            (dim, ia),
            (dim, ia),
            (dim, ia),
            (dim, ia),
            (dim, ia),
        ])
        h0, c0, wxi, whi, wci, wxf, whf, wcf, wxc, whc, wxo, who, wco, bi, bf, bc, bo = wlist
        tmph = h0.dimshuffle('x', 0) + T.zeros((prev_layer.shape[1], dim))
        tmpc = c0.dimshuffle('x', 0) + T.zeros((prev_layer.shape[1], dim))

        def func(x, hp, cp):
            it = sigmoid(T.dot(x, wxi) + T.dot(hp, whi) + cp * wci + bi)
            ft = sigmoid(T.dot(x, wxf) + T.dot(hp, whf) + cp * wcf + bf)
            ot = sigmoid(T.dot(x, wxo) + T.dot(hp, who) + cp * wco + bo)
            ct = ft * cp + it * T.tanh(T.dot(x, wxc) + T.dot(hp, whc) + bc)
            ht = ot * T.tanh(ct)
            return [ht, ct]
        [layer, cl], _ = scan(fn=func, 
                        sequences=prev_layer, 
                        outputs_info=[tmph, tmpc],
                        # truncate_gradient=20
                        )
        return layer, wlist, wdlist

    def output_layer(self, prev_layer, prev_dim, dim):
        alpha = (prev_dim + dim) ** 0.5
        wlh = shared(self.rng.randn(prev_dim, dim).astype(config.floatX) / alpha)
        wlh_delta = shared(np.zeros((prev_dim, dim)).astype(config.floatX))
        b = shared(self.rng.randn(dim).astype(config.floatX) / alpha)
        b_delta = shared(np.zeros(dim).astype(config.floatX))

        layer = T.dot(prev_layer, wlh) + b
        return layer, [wlh, b], [wlh_delta, b_delta]

    def __init__(self, Dims = [1, 1], 
                 Eta = 0.03, Drate = 0.999998, Minrate = 0.2, Momentum = 0.9,
                 Batchsize = 128, K = 48):
        self._Dims = Dims
        self._Eta = Eta
        self._Drate = Drate
        self._Minrate = Minrate
        self._Momentum = Momentum
        self._Batchsize = Batchsize
        self._K = K

        self.rng = np.random.RandomState()
        self.srng = RandomStreams()
        self._L = len(self._Dims)
        self._x = T.tensor3('x', config.floatX)
        # self._x = T.TensorType(dtype=config.floatX, broadcastable=())('x')
        # T.addbroadcast(self._x, 1)
        self._y = T.tensor3('y', config.floatX)
        # self._prob = 0.5
        # self._dropout = False
        # self._p = shared(np.array(self._prob).astype(config.floatX))
        self._w = []
        self._w_delta = []
        self._best_w = []
        self._l = [self._x]
        # self._r = []

        for i in range(1, self._L):
            if i == self._L-1:
                layer, wlist, wdeltalist = self.output_layer(self._l[-1], self._Dims[i-1], self._Dims[i])
            else:
                # layer, wlist, wdeltalist = self.lstm_layer(self._l[-1], self._Dims[i-1], self._Dims[i])
                # layer, wlist, wdeltalist = self.recurrent_layer(self._l[-1], self._Dims[i-1], self._Dims[i])
                # layer, wlist, wdeltalist = self.feedforward_layer(self._l[-1], self._Dims[i-1], self._Dims[i])
                layer, wlist, wdeltalist = self.bidirectional_recurrent_layer(self._l[-1], self._Dims[i-1], self._Dims[i])

            self._l.append(layer)
            self._w.extend(wlist)
            self._w_delta.extend(wdeltalist)
            # self._r.append(srng.binomial((self._Dims[i], self._Dims[i+1]), p=self._p, dtype=config.floatX))

        self._h = self._l[-1]

        dotted = T.exp(self._h)
        dsm = T.sum(dotted, axis=2).dimshuffle(0, 1, 'x')
        yln = self._y * T.log(dotted / dsm)

        self._j = - T.sum(yln) / self._x.shape[1]
        self._j_grad_w = []
        for i in range(len(self._w)):
            self._j_grad_w.append(T.grad(self._j, self._w[i], disconnected_inputs='ignore'))

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
        jfunc = function([self._x, self._y], self._j)
        N_test = X.shape[0]

        # TODO Batch_size = 10000
        Batch_size = self._Batchsize
        Batch_num = (N_test + Batch_size - 1) // Batch_size

        tot_j = 0
        for i in range(Batch_num):
            start = Batch_size * i
            end = start + Batch_size
            X_e = X[start:end]
            YY_e = YY[start:end]
            Xt, Yt = self.pad_zero(X_e, YY_e)
            tot_j += jfunc(Xt, Yt) * Xt.shape[1]

        tot_j /= N_test
        return tot_j

    def update_func(self, degrade_rate, min_eta, momentum):
        udlist = []
        for i in range(len(self._w)):
            udlist.append((self._w_delta[i], self._w_delta[i] * momentum - self._eta * self._j_grad_w[i]))
        udlist.append((self._eta, T.max((self._eta * degrade_rate, min_eta))))
        ret = function([self._x, self._y], None, updates=udlist)
        return ret

    def realupdate_func(self):
        udlist = []
        for i in range(len(self._w)):
            udlist.append((self._w[i], self._w[i] + self._w_delta[i]))
        ret = function([], None, updates=udlist)
        return ret

    def predict_func(self):
        return function([self._x], self._h)

    def fit(self, X, YY):
        self.X = X #[x.astype(config.floatX) for x in X]
        self.YY = YY
        self.N_train = self.X.shape[0]

        Eta = self._Eta
        degrade_rate = self._Drate
        min_eta = self._Eta * self._Minrate
        momentum = self._Momentum

        self._eta.set_value(np.asarray(Eta).astype(config.floatX))
        self.ufunc = self.update_func(degrade_rate, min_eta, momentum)
        self.rfunc = self.realupdate_func()
        self.Epoch = 0

        return self

    def run_train(self, N_epoch=1):
        t0 = time.time()
        to_break = False

        Batch_size = self._Batchsize
        Batch_num = (self.N_train + Batch_size - 1) // Batch_size

        for i in range(N_epoch * Batch_num):
            try:
                Bno = i % Batch_num
                if Bno % 10 == 0:
                    print('Batch', Bno+1)
                # for w in self._w:
                    # print(w.get_value())
                b_start = Batch_size * Bno
                b_end = b_start + Batch_size
                X_e = self.X[b_start:b_end]
                YY_e = self.YY[b_start:b_end]
                
                X_mb, Y_mb = self.pad_zero(X_e, YY_e)

                # print(self._w[-1].get_value())
                # f = function([self._x, self._y], self._j_grad_w[-1])
                # print(np.array(f(X_mb, Y_mb)))

                self.ufunc(X_mb, Y_mb)
                self.rfunc()
                if Bno == Batch_num - 1:
                    self.Epoch += 1
                    J = self.target_value(self.X, self.YY)
                    print('Epoch {0}, \tJ = {1:.5f}, \ttime = {2:.3f}, \teta = {3:.5f}'.format(
                        self.Epoch, float(J), float(time.time()-t0), float(self._eta.get_value()))
                    )

                    t0 = time.time()
                    current_j = J

                    if to_break:
                        return False
                    
            except KeyboardInterrupt:
                to_break = True
                return False

        return True

    def predict(self, X, prob=False):
        N_test = X.shape[0]

        # TODO Batch_size = 10000
        Batch_size = self._Batchsize
        Batch_num = (N_test + Batch_size - 1) // Batch_size

        RTIME = 1
        pfunc = self.predict_func()
        ynums = []
        for i in range(RTIME):
            yps = []
            for i in range(Batch_num):
                start = Batch_size * i
                end = start + Batch_size
                Xp = self.pad_zero(X[start:end])
                res = pfunc(Xp)
                yps.append(self.restore(res, X[start:end]))

            yp = np.concatenate(yps, axis=0)
            ynums.append(yp)
        yp = sum(ynums) / len(ynums)

        if prob:
            return yp
        return yp.argmax(axis=1)

    def pad_zero(self, X, Y=None):
        if Y is None:
            maxlen = max(x.shape[0] for x in X)
            rX = []
            for x in X:
                padlen = maxlen - x.shape[0]
                rX.append(np.pad(x, ((0, padlen), (0, 0)), 'constant', constant_values=(0, 0)))
            return np.array(rX).transpose((1, 0, 2)).astype(config.floatX)
        else:
            maxlen = max(x.shape[0] for x in X)
            rX = []
            rY = []
            for x, y in zip(X, Y):
                y2 = LabelToBinary(y, self._K)
                padlen = maxlen - x.shape[0]
                rX.append(np.pad(x, ((0, padlen), (0, 0)), 'constant', constant_values=(0, 0)))
                rY.append(np.pad(y2, ((0, padlen), (0, 0)), 'constant', constant_values=(0, 0)))
            return np.array(rX).transpose((1, 0, 2)).astype(config.floatX), np.array(rY).transpose((1, 0, 2)).astype(config.floatX)

    def restore(self, Y, X):
        Y = Y.transpose((1, 0, 2))
        res = []
        for x, y in zip(X, Y):
            q = y[:x.shape[0],:]
            res.append(q)
        return np.concatenate(res, axis=0)

    def save_best_w(self):
        self._best_w = [w.get_value() for w in self._w]

    def load_best_w(self):
        for i in range(len(self._w)):
            self._w[i].set_value(self._best_w[i])



