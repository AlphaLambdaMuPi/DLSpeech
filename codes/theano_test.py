import theano.tensor as T
from theano import function

def sigmoid(x):
    return 1 / (1 + T.exp(-x))

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)



