import numpy as np

class GenderEncoder():
    def fit(self, X):
        # this dictionary is used to encode the values
        self.forward = {'female':1, 'male':-1}

    def onehot(self, idx, max):
        z = np.zeros(max)
        z[idx] = 1.0
        return z

    def transform(self,X):
        result = [ self.forward[x] for x in X]
        return np.array(result)

if __name__ == "__main__":
    # example usage
    X = np.array(['female', 'male', 'female', 'male'])
    enc = GenderEncoder()
    enc.fit(X)
    print enc.transform(X)