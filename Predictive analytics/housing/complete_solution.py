import pandas as ps
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

data = ps.read_csv("housing.csv", sep = ',')
XY = data.as_matrix()

def split_data(XY):
    # get input / output arrays
    X = XY[:, :-1]
    Y = XY[:, -1]

    # separate datasaet into training, testing, validation parts:
    # [ training: 60%               | validtn.: 20% | testing: 20% ]
    tr_ix = int(len(X) * 0.6)
    vl_ix = tr_ix + int(len(X) * 0.2)

    X, Xv, Xt = X[:tr_ix], X[tr_ix:vl_ix], X[vl_ix:]
    Y, Yv, Yt = Y[:tr_ix], Y[tr_ix:vl_ix], Y[vl_ix:]

    # perform data normalization
    sc = StandardScaler()
    sc.fit(X)

    X = sc.transform(X)
    Xv = sc.transform(Xv)
    Xt = sc.transform(Xt)

    return X, Y, Xv, Yv, Xt, Yt


def train_linear_model(XY, C = None):

    X, Y, Xv, Yv, Xt, Yt = split_data(XY)

    if C is None:
        alpha = 0.0
    else:
        alpha = 1.0 / C

    model = Lasso(alpha=alpha)

    XXv = np.concatenate([X, Xv])
    YYv = np.concatenate([Y, Yv])

    model.fit(XXv, YYv)
    return model.score(Xt, Yt), model.coef_

def train_knn_model(XY):

    X, Y, Xv, Yv, Xt, Yt = split_data(XY)

    acc_best = -10

    K, ACC = [], []

    for k in range(1, 100):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X, Y)
        acc = knn.score(Xv, Yv)

        K.append(k)
        ACC.append(acc)

        if acc > acc_best:
            knn_k = k
            acc_best = acc

    plt.scatter(K, ACC)
    plt.show()

    XXv = np.concatenate([X, Xv])
    YYv = np.concatenate([Y, Yv])

    knn = KNeighborsRegressor(n_neighbors=knn_k)
    knn.fit(XXv, YYv)
    return knn.score(Xt, Yt)

for c in [None, 0.01, 1.0, 100.0]:
    print ""
    print "<<< Complexity " + str(c) + " >>>"
    acc, coef = train_linear_model(XY, c)
    print "Test R^2:", acc

    # print the data for comparison
    for c, v in zip( data.columns.values[:-1], coef[:-1]):
        print c,v

acc = train_knn_model(XY)
print "Test R^2, KNN:", acc
