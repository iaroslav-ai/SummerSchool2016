import numpy as np
import theano
import theano.tensor as T

N = 20000
feats = 32
neurons = 64

X = np.random.randn(N, feats)

wf = np.random.randn(feats, 1)

Y = np.sin( np.dot(X, wf) * 0.3 )

X = X.astype("float32")
Y = Y.astype("float32")

X, Xv = X[:N/2], X[N/2:]
Y, Yv = Y[:N/2], Y[N/2:]

theano.config.floatX = 'float32'
theano.config.compute_test_value = "warn"

# inputs and outputs of function
x = T.matrix("x")
y = T.matrix("y")

# parameters of the neural net
W = theano.shared(np.random.randn(feats, neurons).astype('float32')*0.1, name="W")
b = theano.shared(np.random.randn(neurons).astype('float32')*0.1, name="b")

#x.tag.test_value = X
#y.tag.test_value = Y

#print w.get_value(), b.get_value()

yp = T.dot(x, W) + b

yp = T.tanh(yp)

g = theano.shared(np.random.randn(neurons, 1).astype('float32')*0.1, name="g")

yp = T.dot(yp, g)

loss = T.mean((yp - y) ** 2) # The cost to optimize
gw, gb, gg = T.grad(loss, [W, b, g])

alpha = 0.01

# Compile expressions to functions
train = theano.function(
            inputs=[x, y],
            outputs=[yp, loss],
            updates={W: W - alpha * gw, b: b - alpha * gb, g: g - alpha*gg},
            name="train")

evl = theano.function(inputs=[x,y],
                      outputs=loss,
                      name="evaluation")

predict = theano.function(inputs=[x], outputs=yp,
                          name="predict")

import time

start = time.time()

for i in range(100000):

    """
    Xs = X
    Ys = Y
    """
    I = np.random.choice(range(len(X)), size=128)
    Xs = X[I]
    Ys = Y[I]

    if i % 100 == 0:
        pred, tr_loss = train(Xs, Ys)
        val_loss = evl(Xv, Yv)

        print tr_loss, val_loss

print "Total time", time.time() - start

print predict(Xv)
