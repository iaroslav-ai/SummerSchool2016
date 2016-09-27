import numpy as np

class NanMeanImputer():
    def __init__(self):
        pass

    def fit(self, X):
        # choose mean of all that is not nan
        # casting to float is needed to make the isnan funciton work
        I = np.logical_not(np.isnan(X.astype(float)))
        self.mean = np.mean(X[I])

    def transform(self, X):
        # fill mean to all that is nan
        R = np.copy(X).astype(float)
        R[ np.isnan(R) ] = self.mean
        return R

if __name__ == "__main__":
    # data to fit the imputer onto
    X = np.array([1.0, np.nan, 1.0, 0.0, np.nan])
    enc = NanMeanImputer()
    enc.fit(X)

    # column with missing values
    M = np.array([np.nan, 0.0, np.nan, 0.0, 1.0])

    print "before:", M
    print "after:", enc.transform(M)