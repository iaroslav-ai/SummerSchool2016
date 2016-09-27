import pandas as ps
import cPickle as pc
from backend import TitanicSolution
import numpy as np

data = ps.read_csv("titanic.csv", sep = ',')
XY = data.as_matrix()

# feature names for columns [2,4,5,6,7,9]
fn = ['Psclass','Gender','Age','SibSp','Parch','Fare']

# class names
cn = ['Lost', 'Survived']

obj = TitanicSolution('tree', "tree", False)
obj.process(np.copy(XY), fn, cn)

obj = TitanicSolution('tree_pruning', "tree", True)
obj.process(np.copy(XY), fn, cn)

obj = TitanicSolution('svm', "svm")
obj.process(np.copy(XY), fn, cn)
