import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

data = pandas.read_csv("housing.csv")
XY = data.as_matrix()

def preprocess(XY):

    X = XY[:, :-1]
    Y = XY[:, -1]

    tr_ix = int(len(X) * 0.6)
    vl_ix = tr_ix + int(len(X) * 0.2)

    X, Xv, Xt = X[:tr_ix], X[tr_ix:vl_ix], X[vl_ix:]
    Y, Yv, Yt = Y[:tr_ix], Y[tr_ix:vl_ix], Y[vl_ix:]

    sc = StandardScaler()
    sc.fit(X)

    X = sc.transform(X)
    Xv = sc.transform(Xv)
    Xt = sc.transform(Xt)

    return X, Xv, Xt, Y, Yv, Yt

X, Xv, Xt, Y, Yv, Yt = preprocess(XY)

C = 1000.0

model = Lasso(alpha=1.0 / C)
model.fit(X, Y)

print model.score(Xt, Yt)

#coef = model.coef_

# print the data for comparison
#for c, v in zip( data.columns.values[:-1], coef[:-1]):
#    print c,v

best_acc = -10.0

for k in range(1, 100):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, Y)

    val = model.score(Xv, Yv)

    if val > best_acc:
        best_k = k
        best_acc = val

model = KNeighborsRegressor(n_neighbors=best_k)
model.fit(X, Y)

print model.score(Xt, Yt)