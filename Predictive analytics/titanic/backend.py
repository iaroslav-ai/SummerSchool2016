"""
This file contains classes needed for data preprocessing
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
from gender_encoder import GenderEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from imputer import NanMeanImputer
import sklearn
from sklearn.tree import DecisionTreeClassifier
from vis_tree import render_data, render_sklearn_tree

def instantiate_classifier(type, params):
    if type == 'boosting':
        return AdaBoostClassifier(**params)
    elif type == 'svm':
        return SVC(**params)
    elif type == 'tree_clsf':
        return DecisionTreeClassifier(**params)

    raise "unknown type of model"


def train_evaluate_params(X, Y, Xv, Yv, p):
    model = instantiate_classifier(p[0], p[1])
    model.fit(X, Y)
    acc = model.score(Xv, Yv)
    return {'acc': acc, 'p': p}

class TitanicSolution():

    def __init__(self, output_name, model_type = "tree", min_sample_pruning=8):
        self.otp_name = output_name
        self.min_sample_pruning = min_sample_pruning
        self.model_type = model_type

    def encode_X(self,X):

        Rx = []
        for x, e in zip(X.T, self.enc_X):
            Rx.append(x if e is None else e.transform(x))

        Rx = np.column_stack(Rx)
        return Rx

    def impute_X(self,X):

        Rx = []
        for x, imp in zip(X.T, self.imp_X):
            Rx.append(x if imp is None else imp.transform(x))

        Rx = np.column_stack(Rx)
        return Rx

    def normalize_X(self,X):
        return self.scaler.transform(X)

    def prepare_columns(self, XY):
        # this removes some columns and places survived in the last position
        result = XY[:, [2,4,5,6,7,9, 1]]
        return result

    def process(self, XY, fn, cn, needs_encoding = True):
        """
        This creates all the data needed for preprocessing of the census dataset

        :param XY: Contents of CSV file as a numpy array
        :param fn: feature names
        :param cn: class names
        :param needs_encoding: if True, this means that the class works with Titanic dataset.
                               Otherwise, the class is working with the Health data.
        :return: None
        """

        # remove unnecessary columns, put the column to be predicted in front
        if needs_encoding:
            XY = self.prepare_columns(XY)

        X = XY[:, :-1]
        Y = XY[:, -1].astype(float)

        # separate datasaet into training, testing, validation parts:
        # [ training: 60%               | validtn.: 20% | testing: 20% ]
        tr_ix = int(len(X) * 0.6)
        vl_ix = tr_ix + int(len(X) * 0.2)

        X, Xv, Xt = X[:tr_ix], X[tr_ix:vl_ix], X[vl_ix:]
        Y, Yv, Yt = Y[:tr_ix], Y[tr_ix:vl_ix], Y[vl_ix:]

        # perform the imputation of the missing values.
        # train imputer on the training part of the dataset
        self.imp_X = []

        for i in range(X.shape[1]):
            # columns for which it is expected that there are missing values
            if not i in [2] or not needs_encoding:
                self.imp_X.append(None)
            else:
                enc = NanMeanImputer()
                enc.fit(X[:,i])

                self.imp_X.append(enc)

        # perform imputation where necessary
        X = self.impute_X(X)
        Xv = self.impute_X(Xv)
        Xt = self.impute_X(Xt)

        # create one hot enecoding where necessary, using training data
        self.enc_X = []

        for i in range(X.shape[1]):
            # columns for which encoding is required
            if not i in [1] or not needs_encoding:
                self.enc_X.append(None)
            else:
                enc = GenderEncoder()
                enc.fit(X[:,i])

                self.enc_X.append(enc)

        X = self.encode_X(X)
        Xv = self.encode_X(Xv)
        Xt = self.encode_X(Xt)

        # no normalization 4 interpretablility



        # set parameters for training to try
        params = []

        if self.model_type == "tree":

            if self.min_sample_pruning:
                # tree params
                for max_depth in [1,2,4,8,16,32,64]:
                    params.append(['tree_clsf', {
                        'max_depth':max_depth,
                        'min_samples_split':10,
                        'min_samples_leaf':10,
                    }])
            else:
                # tree params
                for max_depth in [1,2,4,8,16,32,64]:
                    params.append(['tree_clsf', {
                        'max_depth':max_depth,
                    }])

        else:
            # create and train predictive model
            for C in  np.logspace(-7, 5, num = 13):
                for gamma in np.logspace(-5, 1, num = 7):
                    params.append(['svm',{
                        "C":C,
                        "gamma":gamma,
                        "kernel":"rbf"
                    }])


        scores = []

        # create, train, evaluate predictive model, synchronus loop
        for p in params:
            # check what you get with selected parameters
            result = train_evaluate_params(X, Y, Xv, Yv, p)
            scores.append(result)
            print result

        # find the best one
        params = max(scores, key = lambda s: s['acc'])
        p = params['p']
        print "best parameters: ", p
        print "best val. accuracy: ", params['acc']

        model = instantiate_classifier(p[0], p[1])
        XXv = np.concatenate([X, Xv])
        YYv = np.concatenate([Y, Yv])

        model.fit(XXv, YYv)

        score = model.score(Xt, Yt)

        self.test_accuracy = score
        self.model = model

        print "test accuracy:", score

        if self.model_type == "tree":
            # render the tree
            render_sklearn_tree(self.model, feature_names=fn, target_names=cn, saveto= self.otp_name + '.pdf')

