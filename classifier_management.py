import itertools
import xlsxwriter as xl
import time

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Classifier:
    def __init__(self, title:str, parameters_list:list, combos:list, headers:list):
        self.__title = title
        self.__parameters_list = parameters_list
        self.__combos = combos
        self.__headers = headers + ['COMPONENT-NO.', '%-ACCURACY','BEST_ACCURACY']

    def get_title(self):
        return self.__title

    def get_headers(self):
        return self.__headers

    def get_combos(self):
        return self.__combos

    def __get_accuracy(self, model, train_X, train_Y, test_X, test_Y):
        model.fit(train_X, train_Y)
        res_model = model.predict(test_X)
        return accuracy_score(test_Y, res_model)


class DTreeModel(Classifier):
    def __init__(self):
        self.__title = 'DTree_'
        self.__parameters_list = [['gini', 'entropy'], ['best', 'random'], [1, 1.0, 'auto', 'sqrt', 'log2', None], ['balanced', None]]
        self.__headers = ['CRITERION', 'SPLITTER', 'MAX_FEATURES', 'CLASS_WEIGHT']
        #self.__combos = list(itertools.product(*self.parameters_list))
        self.__combos = [
            ['gini', 'best', 1, 'balanced'], ['gini', 'best', 1, None], ['gini', 'best', 1.0, 'balanced'], ['gini', 'best', 1.0, None], ['gini', 'best', 'auto', 'balanced'], ['gini', 'best', 'auto', None], ['gini', 'best', 'sqrt', 'balanced'], ['gini', 'best', 'sqrt', None], ['gini', 'best', 'log2', 'balanced'], ['gini', 'best', 'log2', None], ['gini', 'best', None, 'balanced'], ['gini', 'best', None, None], ['gini', 'random', 1, 'balanced'], ['gini', 'random', 1, None], ['gini', 'random', 1.0, 'balanced'], ['gini', 'random', 1.0, None], ['gini', 'random', 'auto', 'balanced'], ['gini', 'random', 'auto', None], ['gini', 'random', 'sqrt', 'balanced'], ['gini', 'random', 'sqrt', None], ['gini', 'random', 'log2', 'balanced'], ['gini', 'random', 'log2', None], ['gini', 'random', None, 'balanced'], ['gini', 'random', None, None], ['entropy', 'best', 1, 'balanced'], ['entropy', 'best', 1, None], ['entropy', 'best', 1.0, 'balanced'], ['entropy', 'best', 1.0, None], ['entropy', 'best', 'auto', 'balanced'], ['entropy', 'best', 'auto', None], ['entropy', 'best', 'sqrt', 'balanced'], ['entropy', 'best', 'sqrt', None], ['entropy', 'best', 'log2', 'balanced'], ['entropy', 'best', 'log2', None], ['entropy', 'best', None, 'balanced'], ['entropy', 'best', None, None], ['entropy', 'random', 1, 'balanced'], ['entropy', 'random', 1, None], ['entropy', 'random', 1.0, 'balanced'], ['entropy', 'random', 1.0, None], ['entropy', 'random', 'auto', 'balanced'], ['entropy', 'random', 'auto', None], ['entropy', 'random', 'sqrt', 'balanced'], ['entropy', 'random', 'sqrt', None], ['entropy', 'random', 'log2', 'balanced'], ['entropy', 'random', 'log2', None], ['entropy', 'random', None, 'balanced'], ['entropy', 'random', None, None]
            ]
        
        super(DTreeModel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        #print('Applying combos >> ', self.__combos[c])
        self.__model = DecisionTreeClassifier(
            criterion=self.__combos[c][0], splitter=self.__combos[c][1], max_features=self.__combos[c][2], class_weight=self.__combos[c][3]
        )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class GaussianNBModel(Classifier):
    def __init__(self):
        self.__title = 'GaussNB_'
        var_smoothing = 0.000000001
        self.__parameters_list = [var_smoothing]
        self.__combos = [[var_smoothing]]
        self.__headers = ['VAR_SMOOTHING']

        super(GaussianNBModel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = gaussnb_model = GaussianNB(var_smoothing=self.__combos[c][0])
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class KNeighborModel(Classifier):
    def __init__(self):
        self.__title = 'KNbr_'
        self.__parameters_list = [['uniform', 'distance'], ['auto', 'ball_tree', 'kd_tree', 'brute']]
        self.__combos = [list(x) for x in list(itertools.product(*self.__parameters_list))]
        self.__headers = ['WEIGHTS', 'ALGORITHM']
        super(KNeighborModel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = KNeighborsClassifier(weights=self.__combos[c][0], algorithm=self.__combos[c][1]
        )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class LDAmodel(Classifier):
    def __init__(self):
        self.__title = 'LDA_'
        solver = ['svd', 'lsqr', 'eigen']
        shrinkage = ['auto', 0.1, 0.25, 0.5, 0.75, 0.99, None]
        store_covariance = [True, False]
        tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
        self.__parameters_list = [solver, shrinkage, store_covariance, tol]
        #combo_list = list(itertools.product(*self.__parameters_list))
        # there are initially total 294 combos
        # after checking 210 combos are successful. model will work on that.
        self.__headers = ['SOLVER', 'SHRINKAGE', 'STORE-COVARIANCE','TOLERANCE']
        self.__combos = [
            ['svd', None, True, 0.1], ['svd', None, True, 0.01], ['svd', None, True, 0.001], ['svd', None, True, 0.0001], ['svd', None, True, 0.2], ['svd', None, True, 0.02], ['svd', None, True, 0.002], ['svd', None, False, 0.1], ['svd', None, False, 0.01], ['svd', None, False, 0.001], ['svd', None, False, 0.0001], ['svd', None, False, 0.2], ['svd', None, False, 0.02], ['svd', None, False, 0.002], ['lsqr', 'auto', True, 0.1], ['lsqr', 'auto', True, 0.01], ['lsqr', 'auto', True, 0.001], ['lsqr', 'auto', True, 0.0001], ['lsqr', 'auto', True, 0.2], ['lsqr', 'auto', True, 0.02], ['lsqr', 'auto', True, 0.002], ['lsqr', 'auto', False, 0.1], ['lsqr', 'auto', False, 0.01], ['lsqr', 'auto', False, 0.001], ['lsqr', 'auto', False, 0.0001], ['lsqr', 'auto', False, 0.2], ['lsqr', 'auto', False, 0.02], ['lsqr', 'auto', False, 0.002], ['lsqr', 0.1, True, 0.1], ['lsqr', 0.1, True, 0.01], ['lsqr', 0.1, True, 0.001], ['lsqr', 0.1, True, 0.0001], ['lsqr', 0.1, True, 0.2], ['lsqr', 0.1, True, 0.02], ['lsqr', 0.1, True, 0.002], ['lsqr', 0.1, False, 0.1], ['lsqr', 0.1, False, 0.01], ['lsqr', 0.1, False, 0.001], ['lsqr', 0.1, False, 0.0001], ['lsqr', 0.1, False, 0.2], ['lsqr', 0.1, False, 0.02], ['lsqr', 0.1, False, 0.002], ['lsqr', 0.25, True, 0.1], ['lsqr', 0.25, True, 0.01], ['lsqr', 0.25, True, 0.001], ['lsqr', 0.25, True, 0.0001], ['lsqr', 0.25, True, 0.2], ['lsqr', 0.25, True, 0.02], ['lsqr', 0.25, True, 0.002], ['lsqr', 0.25, False, 0.1], ['lsqr', 0.25, False, 0.01], ['lsqr', 0.25, False, 0.001], ['lsqr', 0.25, False, 0.0001], ['lsqr', 0.25, False, 0.2], ['lsqr', 0.25, False, 0.02], ['lsqr', 0.25, False, 0.002], ['lsqr', 0.5, True, 0.1], ['lsqr', 0.5, True, 0.01], ['lsqr', 0.5, True, 0.001], ['lsqr', 0.5, True, 0.0001], ['lsqr', 0.5, True, 0.2], ['lsqr', 0.5, True, 0.02], ['lsqr', 0.5, True, 0.002], ['lsqr', 0.5, False, 0.1], ['lsqr', 0.5, False, 0.01], ['lsqr', 0.5, False, 0.001], ['lsqr', 0.5, False, 0.0001], ['lsqr', 0.5, False, 0.2], ['lsqr', 0.5, False, 0.02], ['lsqr', 0.5, False, 0.002], ['lsqr', 0.75, True, 0.1], ['lsqr', 0.75, True, 0.01], ['lsqr', 0.75, True, 0.001], ['lsqr', 0.75, True, 0.0001], ['lsqr', 0.75, True, 0.2], ['lsqr', 0.75, True, 0.02], ['lsqr', 0.75, True, 0.002], ['lsqr', 0.75, False, 0.1], ['lsqr', 0.75, False, 0.01], ['lsqr', 0.75, False, 0.001], ['lsqr', 0.75, False, 0.0001], ['lsqr', 0.75, False, 0.2], ['lsqr', 0.75, False, 0.02], ['lsqr', 0.75, False, 0.002], ['lsqr', 0.99, True, 0.1], ['lsqr', 0.99, True, 0.01], ['lsqr', 0.99, True, 0.001], ['lsqr', 0.99, True, 0.0001], ['lsqr', 0.99, True, 0.2], ['lsqr', 0.99, True, 0.02], ['lsqr', 0.99, True, 0.002], ['lsqr', 0.99, False, 0.1], ['lsqr', 0.99, False, 0.01], ['lsqr', 0.99, False, 0.001], ['lsqr', 0.99, False, 0.0001], ['lsqr', 0.99, False, 0.2], ['lsqr', 0.99, False, 0.02], ['lsqr', 0.99, False, 0.002], ['lsqr', None, True, 0.1], ['lsqr', None, True, 0.01], ['lsqr', None, True, 0.001], ['lsqr', None, True, 0.0001], ['lsqr', None, True, 0.2], ['lsqr', None, True, 0.02], ['lsqr', None, True, 0.002], ['lsqr', None, False, 0.1], ['lsqr', None, False, 0.01], [
                'lsqr', None, False, 0.001], ['lsqr', None, False, 0.0001], ['lsqr', None, False, 0.2], ['lsqr', None, False, 0.02], ['lsqr', None, False, 0.002], ['eigen', 'auto', True, 0.1], ['eigen', 'auto', True, 0.01], ['eigen', 'auto', True, 0.001], ['eigen', 'auto', True, 0.0001], ['eigen', 'auto', True, 0.2], ['eigen', 'auto', True, 0.02], ['eigen', 'auto', True, 0.002], ['eigen', 'auto', False, 0.1], ['eigen', 'auto', False, 0.01], ['eigen', 'auto', False, 0.001], ['eigen', 'auto', False, 0.0001], ['eigen', 'auto', False, 0.2], ['eigen', 'auto', False, 0.02], ['eigen', 'auto', False, 0.002], ['eigen', 0.1, True, 0.1], ['eigen', 0.1, True, 0.01], ['eigen', 0.1, True, 0.001], ['eigen', 0.1, True, 0.0001], ['eigen', 0.1, True, 0.2], ['eigen', 0.1, True, 0.02], ['eigen', 0.1, True, 0.002], ['eigen', 0.1, False, 0.1], ['eigen', 0.1, False, 0.01], ['eigen', 0.1, False, 0.001], ['eigen', 0.1, False, 0.0001], ['eigen', 0.1, False, 0.2], ['eigen', 0.1, False, 0.02], ['eigen', 0.1, False, 0.002], ['eigen', 0.25, True, 0.1], ['eigen', 0.25, True, 0.01], ['eigen', 0.25, True, 0.001], ['eigen', 0.25, True, 0.0001], ['eigen', 0.25, True, 0.2], ['eigen', 0.25, True, 0.02], ['eigen', 0.25, True, 0.002], ['eigen', 0.25, False, 0.1], ['eigen', 0.25, False, 0.01], ['eigen', 0.25, False, 0.001], ['eigen', 0.25, False, 0.0001], ['eigen', 0.25, False, 0.2], ['eigen', 0.25, False, 0.02], ['eigen', 0.25, False, 0.002], ['eigen', 0.5, True, 0.1], ['eigen', 0.5, True, 0.01], ['eigen', 0.5, True, 0.001], ['eigen', 0.5, True, 0.0001], ['eigen', 0.5, True, 0.2], ['eigen', 0.5, True, 0.02], ['eigen', 0.5, True, 0.002], ['eigen', 0.5, False, 0.1], ['eigen', 0.5, False, 0.01], ['eigen', 0.5, False, 0.001], ['eigen', 0.5, False, 0.0001], ['eigen', 0.5, False, 0.2], ['eigen', 0.5, False, 0.02], ['eigen', 0.5, False, 0.002], ['eigen', 0.75, True, 0.1], ['eigen', 0.75, True, 0.01], ['eigen', 0.75, True, 0.001], ['eigen', 0.75, True, 0.0001], ['eigen', 0.75, True, 0.2], ['eigen', 0.75, True, 0.02], ['eigen', 0.75, True, 0.002], ['eigen', 0.75, False, 0.1], ['eigen', 0.75, False, 0.01], ['eigen', 0.75, False, 0.001], ['eigen', 0.75, False, 0.0001], ['eigen', 0.75, False, 0.2], ['eigen', 0.75, False, 0.02], ['eigen', 0.75, False, 0.002], ['eigen', 0.99, True, 0.1], ['eigen', 0.99, True, 0.01], ['eigen', 0.99, True, 0.001], ['eigen', 0.99, True, 0.0001], ['eigen', 0.99, True, 0.2], ['eigen', 0.99, True, 0.02], ['eigen', 0.99, True, 0.002], ['eigen', 0.99, False, 0.1], ['eigen', 0.99, False, 0.01], ['eigen', 0.99, False, 0.001], ['eigen', 0.99, False, 0.0001], ['eigen', 0.99, False, 0.2], ['eigen', 0.99, False, 0.02], ['eigen', 0.99, False, 0.002], ['eigen', None, True, 0.1], ['eigen', None, True, 0.01], ['eigen', None, True, 0.001], ['eigen', None, True, 0.0001], ['eigen', None, True, 0.2], ['eigen', None, True, 0.02], ['eigen', None, True, 0.002], ['eigen', None, False, 0.1], ['eigen', None, False, 0.01], ['eigen', None, False, 0.001], ['eigen', None, False, 0.0001], ['eigen', None, False, 0.2], ['eigen', None, False, 0.02], ['eigen', None, False, 0.002]
        ]
        super(LDAmodel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = LinearDiscriminantAnalysis(
            solver=self.__combos[c][0], shrinkage=self.__combos[c][1],store_covariance=self.__combos[c][2], tol=self.__combos[c][3]
        )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class LogRegModel(Classifier):
    def __init__(self):
        self.__title = 'LogReg_'
        penalty = ['l1', 'l2', 'elasticnet', 'none']
        dual = [False, True]
        tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
        fit_intercept = [False, True]
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        multiclass = ['auto', 'ovr', 'multinomial']
        warm = [True, False]
        self.__parameters_list = [penalty, dual, tol, fit_intercept, solver, multiclass, warm]
        #combo_list = list(itertools.product(*parameters_list))
        # after checking 924 combos were successful. model will work on that.
        # now this has 336 combos after some more filtering
        self.__headers = ['PENALTY', 'DUAL', 'TOLERANCE', 'FIT_INTERCEPT', 'SOLVER', 'MULTICLASS', 'WARM_START']
        self.__combos = [
            ['l1', False, 0.1, False, 'liblinear', 'auto', True], ['l1', False, 0.1, False, 'liblinear', 'ovr', True], ['l1', False, 0.1, False, 'saga', 'auto', True], ['l1', False, 0.1, False, 'saga', 'ovr', True], ['l1', False, 0.1, True, 'liblinear', 'auto', True], ['l1', False, 0.1, True, 'liblinear', 'ovr', True], ['l1', False, 0.1, True, 'saga', 'auto', True], ['l1', False, 0.1, True, 'saga', 'ovr', True], ['l1', False, 0.01, False, 'liblinear', 'auto', True], ['l1', False, 0.01, False, 'liblinear', 'ovr', True], ['l1', False, 0.01, False, 'saga', 'auto', True], ['l1', False, 0.01, False, 'saga', 'ovr', True], ['l1', False, 0.01, True, 'liblinear', 'auto', True], ['l1', False, 0.01, True, 'liblinear', 'ovr', True], ['l1', False, 0.01, True, 'saga', 'auto', True], ['l1', False, 0.01, True, 'saga', 'ovr', True], ['l1', False, 0.001, False, 'liblinear', 'auto', True], ['l1', False, 0.001, False, 'liblinear', 'ovr', True], ['l1', False, 0.001, False, 'saga', 'auto', True], ['l1', False, 0.001, False, 'saga', 'ovr', True], ['l1', False, 0.001, True, 'liblinear', 'auto', True], ['l1', False, 0.001, True, 'liblinear', 'ovr', True], ['l1', False, 0.001, True, 'saga', 'auto', True], ['l1', False, 0.001, True, 'saga', 'ovr', True], ['l1', False, 0.0001, False, 'liblinear', 'auto', True], ['l1', False, 0.0001, False, 'liblinear', 'ovr', True], ['l1', False, 0.0001, False, 'saga', 'auto', True], ['l1', False, 0.0001, False, 'saga', 'ovr', True], ['l1', False, 0.0001, True, 'liblinear', 'auto', True], ['l1', False, 0.0001, True, 'liblinear', 'ovr', True], ['l1', False, 0.0001, True, 'saga', 'auto', True], ['l1', False, 0.0001, True, 'saga', 'ovr', True], ['l1', False, 0.2, False, 'liblinear', 'auto', True], ['l1', False, 0.2, False, 'liblinear', 'ovr', True], ['l1', False, 0.2, False, 'saga', 'auto', True], ['l1', False, 0.2, False, 'saga', 'ovr', True], ['l1', False, 0.2, True, 'liblinear', 'auto', True], ['l1', False, 0.2, True, 'liblinear', 'ovr', True], ['l1', False, 0.2, True, 'saga', 'auto', True], ['l1', False, 0.2, True, 'saga', 'ovr', True], ['l1', False, 0.02, False, 'liblinear', 'auto', True], ['l1', False, 0.02, False, 'liblinear', 'ovr', True], ['l1', False, 0.02, False, 'saga', 'auto', True], ['l1', False, 0.02, False, 'saga', 'ovr', True], ['l1', False, 0.02, True, 'liblinear', 'auto', True], ['l1', False, 0.02, True, 'liblinear', 'ovr', True], ['l1', False, 0.02, True, 'saga', 'auto', True], ['l1', False, 0.02, True, 'saga', 'ovr', True], ['l1', False, 0.002, False, 'liblinear', 'auto', True], ['l1', False, 0.002, False, 'liblinear', 'ovr', True], ['l1', False, 0.002, False, 'saga', 'auto', True], ['l1', False, 0.002, False, 'saga', 'ovr', True], ['l1', False, 0.002, True, 'liblinear', 'auto', True], ['l1', False, 0.002, True, 'liblinear', 'ovr', True], ['l1', False, 0.002, True, 'saga', 'auto', True], ['l1', False, 0.002, True, 'saga', 'ovr', True], ['l2', False, 0.1, False, 'newton-cg', 'auto', True], ['l2', False, 0.1, False, 'newton-cg', 'ovr', True], ['l2', False, 0.1, False, 'lbfgs', 'auto', True], ['l2', False, 0.1, False, 'lbfgs', 'ovr', True], ['l2', False, 0.1, False, 'liblinear', 'auto', True], ['l2', False, 0.1, False, 'liblinear', 'ovr', True], ['l2', False, 0.1, False, 'sag', 'auto', True], ['l2', False, 0.1, False, 'sag', 'ovr', True], ['l2', False, 0.1, False, 'saga', 'auto', True], ['l2', False, 0.1, False, 'saga', 'ovr', True], ['l2', False, 0.1, True, 'newton-cg', 'auto', True], ['l2', False, 0.1, True, 'newton-cg', 'ovr', True], ['l2', False, 0.1, True, 'lbfgs', 'auto', True], ['l2', False, 0.1, True, 'lbfgs', 'ovr', True], ['l2', False, 0.1, True, 'liblinear', 'auto', True], ['l2', False, 0.1, True, 'liblinear', 'ovr', True], ['l2', False, 0.1, True, 'sag', 'auto', True], ['l2', False, 0.1, True, 'sag', 'ovr', True], ['l2', False, 0.1, True, 'saga', 'auto', True], ['l2', False, 0.1, True, 'saga', 'ovr', True], ['l2', False, 0.01, False, 'newton-cg', 'auto', True], ['l2', False, 0.01, False, 'newton-cg', 'ovr', True], ['l2', False, 0.01, False, 'lbfgs', 'auto', True], ['l2', False, 0.01, False, 'lbfgs', 'ovr', True], ['l2', False, 0.01, False, 'liblinear', 'auto', True], ['l2', False, 0.01, False, 'liblinear', 'ovr', True], ['l2', False, 0.01, False, 'sag', 'auto', True], ['l2', False, 0.01, False, 'sag', 'ovr', True], ['l2', False, 0.01, False, 'saga', 'auto', True], ['l2', False, 0.01, False, 'saga', 'ovr', True], ['l2', False, 0.01, True, 'newton-cg', 'auto', True], ['l2', False, 0.01, True, 'newton-cg', 'ovr', True], ['l2', False, 0.01, True, 'lbfgs', 'auto', True], ['l2', False, 0.01, True, 'lbfgs', 'ovr', True], ['l2', False, 0.01, True, 'liblinear', 'auto', True], ['l2', False, 0.01, True, 'liblinear', 'ovr', True], ['l2', False, 0.01, True, 'sag', 'auto', True], ['l2', False, 0.01, True, 'sag', 'ovr', True], ['l2', False, 0.01, True, 'saga', 'auto', True], ['l2', False, 0.01, True, 'saga', 'ovr', True], ['l2', False, 0.001, False, 'newton-cg', 'auto', True], ['l2', False, 0.001, False, 'newton-cg', 'ovr', True], ['l2', False, 0.001, False, 'lbfgs', 'auto', True], ['l2', False, 0.001, False, 'lbfgs', 'ovr', True], ['l2', False, 0.001, False, 'liblinear', 'auto', True], ['l2', False, 0.001, False, 'liblinear', 'ovr', True], ['l2', False, 0.001, False, 'sag', 'auto', True], ['l2', False, 0.001, False, 'sag', 'ovr', True], ['l2', False, 0.001, False, 'saga', 'auto', True], ['l2', False, 0.001, False, 'saga', 'ovr', True], ['l2', False, 0.001, True, 'newton-cg', 'auto', True], ['l2', False, 0.001, True, 'newton-cg', 'ovr', True], ['l2', False, 0.001, True, 'lbfgs', 'auto', True], ['l2', False, 0.001, True, 'lbfgs', 'ovr', True], ['l2', False, 0.001, True, 'liblinear', 'auto', True], ['l2', False, 0.001, True, 'liblinear', 'ovr', True], ['l2', False, 0.001, True, 'sag', 'auto', True], ['l2', False, 0.001, True, 'sag', 'ovr', True], ['l2', False, 0.001, True, 'saga', 'auto', True], ['l2', False, 0.001, True, 'saga', 'ovr', True], ['l2', False, 0.0001, False, 'newton-cg', 'auto', True], ['l2', False, 0.0001, False, 'newton-cg', 'ovr', True], ['l2', False, 0.0001, False, 'lbfgs', 'auto', True], ['l2', False, 0.0001, False, 'lbfgs', 'ovr', True], ['l2', False, 0.0001, False, 'liblinear', 'auto', True], ['l2', False, 0.0001, False, 'liblinear', 'ovr', True], ['l2', False, 0.0001, False, 'sag', 'auto', True], ['l2', False, 0.0001, False, 'sag', 'ovr', True], ['l2', False, 0.0001, False, 'saga', 'auto', True], ['l2', False, 0.0001, False, 'saga', 'ovr', True], ['l2', False, 0.0001, True, 'newton-cg', 'auto', True], ['l2', False, 0.0001, True, 'newton-cg', 'ovr', True], ['l2', False, 0.0001, True, 'lbfgs', 'auto', True], ['l2', False, 0.0001, True, 'lbfgs', 'ovr', True], ['l2', False, 0.0001, True, 'liblinear', 'auto', True], ['l2', False, 0.0001, True, 'liblinear', 'ovr', True], ['l2', False, 0.0001, True, 'sag', 'auto', True], ['l2', False, 0.0001, True, 'sag', 'ovr', True], ['l2', False, 0.0001, True, 'saga', 'auto', True], ['l2', False, 0.0001, True, 'saga', 'ovr', True], ['l2', False, 0.2, False, 'newton-cg', 'auto', True], ['l2', False, 0.2, False, 'newton-cg', 'ovr', True], ['l2', False, 0.2, False, 'lbfgs', 'auto', True], ['l2', False, 0.2, False, 'lbfgs', 'ovr', True], ['l2', False, 0.2, False, 'liblinear', 'auto', True], ['l2', False, 0.2, False, 'liblinear', 'ovr', True], ['l2', False, 0.2, False, 'sag', 'auto', True], ['l2', False, 0.2, False, 'sag', 'ovr', True], ['l2', False, 0.2, False, 'saga', 'auto', True], ['l2', False, 0.2, False, 'saga', 'ovr', True], ['l2', False, 0.2, True, 'newton-cg', 'auto', True], ['l2', False, 0.2, True, 'newton-cg', 'ovr', True], ['l2', False, 0.2, True, 'lbfgs', 'auto', True], ['l2', False, 0.2, True, 'lbfgs', 'ovr', True], ['l2', False, 0.2, True, 'liblinear', 'auto', True], ['l2', False, 0.2, True, 'liblinear', 'ovr', True], ['l2', False, 0.2, True, 'sag', 'auto', True], ['l2', False, 0.2, True, 'sag', 'ovr', True], ['l2', False, 0.2, True, 'saga', 'auto', True], ['l2', False, 0.2, True, 'saga', 'ovr', True], ['l2', False, 0.02, False, 'newton-cg', 'auto', True], ['l2', False, 0.02, False, 'newton-cg', 'ovr', True], ['l2', False, 0.02, False, 'lbfgs', 'auto', True], ['l2', False, 0.02, False, 'lbfgs', 'ovr', True], ['l2', False, 0.02, False, 'liblinear', 'auto', True], ['l2', False, 0.02, False, 'liblinear', 'ovr', True], ['l2', False, 0.02, False, 'sag', 'auto', True], ['l2', False, 0.02, False, 'sag', 'ovr', True], ['l2', False, 0.02, False, 'saga', 'auto', True], ['l2', False, 0.02, False, 'saga', 'ovr', True], ['l2', False, 0.02, True, 'newton-cg', 'auto', True], ['l2', False, 0.02, True, 'newton-cg', 'ovr', True], ['l2', False, 0.02, True, 'lbfgs', 'auto', True], ['l2', False, 0.02, True, 'lbfgs', 'ovr', True], ['l2', False, 0.02, True, 'liblinear', 'auto', True], ['l2', False, 0.02, True, 'liblinear', 'ovr', True], ['l2', False, 0.02, True, 'sag', 'auto', True], ['l2', False, 0.02, True, 'sag', 'ovr', True], ['l2', False, 0.02, True, 'saga', 'auto', True], ['l2', False, 0.02, True, 'saga', 'ovr', True], ['l2', False, 0.002, False, 'newton-cg', 'auto', True], ['l2', False, 0.002, False, 'newton-cg', 'ovr', True], ['l2', False, 0.002, False, 'lbfgs', 'auto', True], ['l2', False, 0.002, False, 'lbfgs', 'ovr', True], ['l2', False, 0.002, False, 'liblinear', 'auto', True], ['l2', False, 0.002, False, 'liblinear', 'ovr', True], ['l2', False, 0.002, False, 'sag', 'auto', True], ['l2', False, 0.002, False, 'sag', 'ovr', True], ['l2', False, 0.002, False, 'saga', 'auto', True], ['l2', False, 0.002, False, 'saga', 'ovr', True], ['l2', False, 0.002, True, 'newton-cg', 'auto', True], ['l2', False, 0.002, True, 'newton-cg', 'ovr', True], ['l2', False, 0.002, True, 'lbfgs', 'auto', True], ['l2', False, 0.002, True, 'lbfgs', 'ovr', True], ['l2', False, 0.002, True, 'liblinear', 'auto', True], ['l2', False, 0.002, True, 'liblinear', 'ovr', True], ['l2', False, 0.002, True, 'sag', 'auto', True], ['l2', False, 0.002, True, 'sag', 'ovr', True], ['l2', False, 0.002, True, 'saga', 'auto', True], ['l2', False, 0.002, True, 'saga', 'ovr', True], ['l2', True, 0.1, False, 'liblinear', 'auto', True], ['l2', True, 0.1, False, 'liblinear', 'ovr', True], ['l2', True, 0.1, True, 'liblinear', 'auto', True], ['l2', True, 0.1, True, 'liblinear', 'ovr', True], ['l2', True, 0.01, False, 'liblinear', 'auto', True], ['l2', True, 0.01, False, 'liblinear', 'ovr', True], ['l2', True, 0.01, True, 'liblinear', 'auto', True], ['l2', True, 0.01, True, 'liblinear', 'ovr', True], ['l2', True, 0.001, False, 'liblinear', 'auto', True], ['l2', True, 0.001, False, 'liblinear', 'ovr', True], ['l2', True, 0.001, True, 'liblinear', 'auto', True], ['l2', True, 0.001, True, 'liblinear', 'ovr', True], ['l2', True, 0.0001, False, 'liblinear', 'auto', True], ['l2', True, 0.0001, False, 'liblinear', 'ovr', True], ['l2', True, 0.0001, True, 'liblinear', 'auto', True], ['l2', True, 0.0001, True, 'liblinear', 'ovr', True], ['l2', True, 0.2, False, 'liblinear', 'auto', True], ['l2', True, 0.2, False, 'liblinear', 'ovr', True], ['l2', True, 0.2, True, 'liblinear', 'auto', True], ['l2', True, 0.2, True, 'liblinear', 'ovr', True], ['l2', True, 0.02, False, 'liblinear', 'auto', True], ['l2', True, 0.02, False, 'liblinear', 'ovr', True], ['l2', True, 0.02, True, 'liblinear', 'auto', True], ['l2', True, 0.02, True, 'liblinear', 'ovr', True], ['l2', True, 0.002, False, 'liblinear', 'auto', True], ['l2', True, 0.002, False, 'liblinear', 'ovr', True], ['l2', True, 0.002, True, 'liblinear', 'auto', True], ['l2', True, 0.002, True, 'liblinear', 'ovr', True], ['none', False, 0.1, False, 'newton-cg', 'auto', True], ['none', False, 0.1, False, 'newton-cg', 'ovr', True], ['none', False, 0.1, False, 'lbfgs', 'auto', True], ['none', False, 0.1, False, 'lbfgs', 'ovr', True], ['none', False, 0.1, False, 'sag', 'auto', True], ['none', False, 0.1, False, 'sag', 'ovr', True], ['none', False, 0.1, False, 'saga', 'auto', True], ['none', False, 0.1, False, 'saga', 'ovr', True], ['none', False, 0.1, True, 'newton-cg', 'auto', True], ['none', False, 0.1, True, 'newton-cg', 'ovr', True], ['none', False, 0.1, True, 'lbfgs', 'auto', True], ['none', False, 0.1, True, 'lbfgs', 'ovr', True], ['none', False, 0.1, True, 'sag', 'auto', True], ['none', False, 0.1, True, 'sag', 'ovr', True], ['none', False, 0.1, True, 'saga', 'auto', True], ['none', False, 0.1, True, 'saga', 'ovr', True], ['none', False, 0.01, False, 'newton-cg', 'auto', True], ['none', False, 0.01, False, 'newton-cg', 'ovr', True], ['none', False, 0.01, False, 'lbfgs', 'auto', True], ['none', False, 0.01, False, 'lbfgs', 'ovr', True], ['none', False, 0.01, False, 'sag', 'auto', True], ['none', False, 0.01, False, 'sag', 'ovr', True], ['none', False, 0.01, False, 'saga', 'auto', True], ['none', False, 0.01, False, 'saga', 'ovr', True], ['none', False, 0.01, True, 'newton-cg', 'auto', True], ['none', False, 0.01, True, 'newton-cg', 'ovr', True], ['none', False, 0.01, True, 'lbfgs', 'auto', True], ['none', False, 0.01, True, 'lbfgs', 'ovr', True], ['none', False, 0.01, True, 'sag', 'auto', True], ['none', False, 0.01, True, 'sag', 'ovr', True], ['none', False, 0.01, True, 'saga', 'auto', True], ['none', False, 0.01, True, 'saga', 'ovr', True], ['none', False, 0.001, False, 'newton-cg', 'auto', True], ['none', False, 0.001, False, 'newton-cg', 'ovr', True], ['none', False, 0.001, False, 'lbfgs', 'auto', True], ['none', False, 0.001, False, 'lbfgs', 'ovr', True], ['none', False, 0.001, False, 'sag', 'auto', True], ['none', False, 0.001, False, 'sag', 'ovr', True], ['none', False, 0.001, False, 'saga', 'auto', True], ['none', False, 0.001, False, 'saga', 'ovr', True], ['none', False, 0.001, True, 'newton-cg', 'auto', True], ['none', False, 0.001, True, 'newton-cg', 'ovr', True], ['none', False, 0.001, True, 'lbfgs', 'auto', True], ['none', False, 0.001, True, 'lbfgs', 'ovr', True], ['none', False, 0.001, True, 'sag', 'auto', True], ['none', False, 0.001, True, 'sag', 'ovr', True], ['none', False, 0.001, True, 'saga', 'auto', True], ['none', False, 0.001, True, 'saga', 'ovr', True], ['none', False, 0.0001, False, 'newton-cg', 'auto', True], ['none', False, 0.0001, False, 'newton-cg', 'ovr', True], ['none', False, 0.0001, False, 'lbfgs', 'auto', True], ['none', False, 0.0001, False, 'lbfgs', 'ovr', True], ['none', False, 0.0001, False, 'sag', 'auto', True], ['none', False, 0.0001, False, 'sag', 'ovr', True], ['none', False, 0.0001, False, 'saga', 'auto', True], ['none', False, 0.0001, False, 'saga', 'ovr', True], ['none', False, 0.0001, True, 'newton-cg', 'auto', True], ['none', False, 0.0001, True, 'newton-cg', 'ovr', True], ['none', False, 0.0001, True, 'lbfgs', 'auto', True], ['none', False, 0.0001, True, 'lbfgs', 'ovr', True], ['none', False, 0.0001, True, 'sag', 'auto', True], ['none', False, 0.0001, True, 'sag', 'ovr', True], ['none', False, 0.0001, True, 'saga', 'auto', True], ['none', False, 0.0001, True, 'saga', 'ovr', True], ['none', False, 0.2, False, 'newton-cg', 'auto', True], ['none', False, 0.2, False, 'newton-cg', 'ovr', True], ['none', False, 0.2, False, 'lbfgs', 'auto', True], ['none', False, 0.2, False, 'lbfgs', 'ovr', True], ['none', False, 0.2, False, 'sag', 'auto', True], ['none', False, 0.2, False, 'sag', 'ovr', True], ['none', False, 0.2, False, 'saga', 'auto', True], ['none', False, 0.2, False, 'saga', 'ovr', True], ['none', False, 0.2, True, 'newton-cg', 'auto', True], ['none', False, 0.2, True, 'newton-cg', 'ovr', True], ['none', False, 0.2, True, 'lbfgs', 'auto', True], ['none', False, 0.2, True, 'lbfgs', 'ovr', True], ['none', False, 0.2, True, 'sag', 'auto', True], ['none', False, 0.2, True, 'sag', 'ovr', True], ['none', False, 0.2, True, 'saga', 'auto', True], ['none', False, 0.2, True, 'saga', 'ovr', True], ['none', False, 0.02, False, 'newton-cg', 'auto', True], ['none', False, 0.02, False, 'newton-cg', 'ovr', True], ['none', False, 0.02, False, 'lbfgs', 'auto', True], ['none', False, 0.02, False, 'lbfgs', 'ovr', True], ['none', False, 0.02, False, 'sag', 'auto', True], ['none', False, 0.02, False, 'sag', 'ovr', True], ['none', False, 0.02, False, 'saga', 'auto', True], ['none', False, 0.02, False, 'saga', 'ovr', True], ['none', False, 0.02, True, 'newton-cg', 'auto', True], ['none', False, 0.02, True, 'newton-cg', 'ovr', True], ['none', False, 0.02, True, 'lbfgs', 'auto', True], ['none', False, 0.02, True, 'lbfgs', 'ovr', True], ['none', False, 0.02, True, 'sag', 'auto', True], ['none', False, 0.02, True, 'sag', 'ovr', True], ['none', False, 0.02, True, 'saga', 'auto', True], ['none', False, 0.02, True, 'saga', 'ovr', True], ['none', False, 0.002, False, 'newton-cg', 'auto', True], ['none', False, 0.002, False, 'newton-cg', 'ovr', True], ['none', False, 0.002, False, 'lbfgs', 'auto', True], ['none', False, 0.002, False, 'lbfgs', 'ovr', True], ['none', False, 0.002, False, 'sag', 'auto', True], ['none', False, 0.002, False, 'sag', 'ovr', True], ['none', False, 0.002, False, 'saga', 'auto', True], ['none', False, 0.002, False, 'saga', 'ovr', True], ['none', False, 0.002, True, 'newton-cg', 'auto', True], ['none', False, 0.002, True, 'newton-cg', 'ovr', True], ['none', False, 0.002, True, 'lbfgs', 'auto', True], ['none', False, 0.002, True, 'lbfgs', 'ovr', True], ['none', False, 0.002, True, 'sag', 'auto', True], ['none', False, 0.002, True, 'sag', 'ovr', True], ['none', False, 0.002, True, 'saga', 'auto', True], ['none', False, 0.002, True, 'saga', 'ovr', True]
        ]
        super(LogRegModel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = LogisticRegression(
            penalty=self.__combos[c][0], dual=self.__combos[c][1], tol=self.__combos[c][2],
            fit_intercept=self.__combos[c][3], max_iter=1000, random_state=0,
            solver=self.__combos[c][4], multi_class=self.__combos[c][5], warm_start=self.__combos[c][6]
        )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class RForestModel(Classifier):
    def __init__(self):
        self.__title = 'RForest_'
        self.__parameters_list = [ ['uniform', 'distance'], ['auto', 'ball_tree', 'kd_tree', 'brute'] ]
        self.__combos = [
            ['gini', 'auto', True, True, True], ['gini', 'auto', True, True, False], ['gini', 'auto', True, False, True], ['gini', 'auto', True, False, False], ['gini', 'auto', False, False, True], ['gini', 'auto', False, False, False], ['gini', 'sqrt', True, True, True], ['gini', 'sqrt', True, True, False], ['gini', 'sqrt', True, False, True], ['gini', 'sqrt', True, False, False], ['gini', 'sqrt', False, False, True], ['gini', 'sqrt', False, False, False], ['gini', 'log2', True, True, True], ['gini', 'log2', True, True, False], ['gini', 'log2', True, False, True], ['gini', 'log2', True, False, False], ['gini', 'log2', False, False, True], ['gini', 'log2', False, False, False], ['entropy', 'auto', True,True, True], ['entropy', 'auto', True, True, False], ['entropy', 'auto', True, False, True], ['entropy', 'auto', True, False, False], ['entropy', 'auto', False, False, True], ['entropy', 'auto', False, False, False], ['entropy', 'sqrt', True, True, True], ['entropy', 'sqrt', True, True, False], ['entropy', 'sqrt', True, False, True], ['entropy', 'sqrt', True, False, False], ['entropy', 'sqrt', False, False, True], ['entropy', 'sqrt', False, False, False], ['entropy', 'log2', True, True, True], ['entropy', 'log2', True, True, False], ['entropy', 'log2', True, False, True], ['entropy', 'log2', True, False, False], ['entropy', 'log2', False, False, True], ['entropy', 'log2', False, False, False]
        ]
        self.__headers = ['CRITERION', 'MAX_FEATURES', 'BOOTSTRAP', 'OOB_SCORE','WARM_START']
        super(RForestModel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = RandomForestClassifier(
            criterion=self.__combos[c][0], max_features=self.__combos[c][1], bootstrap=self.__combos[c][2],
            oob_score=self.__combos[c][3], warm_start=self.__combos[c][4]
            )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class SVCmodel(Classifier):
    def __init__(self):
        self.__title = 'SVC_'
        self.__parameters_list = [['linear', 'poly', 'rbf', 'sigmoid'], ['ovo', 'ovr']]
        self.__combos = [list(x) for x in list(itertools.product(*self.__parameters_list))]
        self.__headers = ['KERNEL', 'DECISION_FUNC_SHAPE',]
        super(SVCmodel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = SVC(kernel=self.__combos[c][0], decision_function_shape=self.__combos[c][1])
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


def prepare_data(data_path):
    all_data = np.load(data_path, allow_pickle=True)
    shape = all_data.shape
    print('Data shape >> ', shape)
    all_X = all_data[:, :-1]
    all_Y = all_data[:, -1]
    print('Data distribution complete.')
    return all_X, all_Y, shape[0]-1


def applyPCA(feature, no_comp):
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components=no_comp)
    pcomp = pca.fit_transform(X)
    return pcomp


def train_model(df, headers, classifier, X, Y, serial=1, doCompo=False):
    
    combo_list = classifier.get_combos()
    number_of_combos = len(combo_list)

    print('Processing compo #', serial)
    x = X
    if doCompo:
        x = applyPCA(X, serial)
        print('PCA successfully applied for component #%d' % (serial+1))

    success = 0
    best_score = 0
    fail = 0
    _scores = []
    n_samples = 162

    ###
    #scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
    #x = scaler.fit_transform(x)
    ###

    train_X, test_X, train_Y, test_Y = train_test_split(x, Y, test_size=0.2)

    for c in range(number_of_combos):
        print('Entering combo #', c+1)
        try:
            score_model = classifier.make_model(c, train_X, train_Y, test_X, test_Y)
            print('Compo #{} - Combo #{} - #{} Combos Successful!\nScore: {}'.format(serial, c+1, success+1, score_model))
            success += 1
            if score_model > best_score:
                print('New highest accuracy:', score_model, '>', best_score)
                print(combo_list[c])
                best_score = score_model
                _scores.append(best_score)
                parameters = combo_list[c] + [serial, float( "{:.2f}".format(best_score * 100)) , best_score]
                d = {}
                for h, p in zip(headers, parameters):
                    d[h] = p
                df = df.append(d, ignore_index=True)
        except:
            print('Compo #', serial, ' - Combo failed at #', c+1)
            fail += 1
        print('Exiting compo #%d - combo #%d\n' % (serial, c+1))
    print('Compo %d - all done.' % serial)
    print('Total combinations: ', number_of_combos)
    print('Total success: ', success)
    print('Total failure:', fail)
    return df


def classify_glcm(model, path):
    X, Y, limit = prepare_data(path)
    line = 1
    scores = []
    
    headers = model.get_headers()
    df = pd.DataFrame(columns=headers)
    
    for serial in range(1, limit):
        df = train_model(df, headers, model, X, Y, serial, True)
        print('Serial #', serial, 'done.')
    return df
    

def save_excel(model, dfs, excel_loc):
    title = model.get_title()
    save_as = excel_loc + f"{title}-clahe.xlsx"
    writer = pd.ExcelWriter(save_as, engine='xlsxwriter')
    counter = 1
    for df in dfs:
        df = df.sort_values(by=['%-ACCURACY'], ascending=False)
        df.to_excel(writer, sheet_name=f'Sheet{counter}', index=False)
        counter += 1
    writer.save()
    return

if __name__ == "__main__":    
    start_time = time.time()

    file_loc = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\clahe_glcm_54.npy"
    excel_loc = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\excels\\80-20-clahe\\'

    #model = DTreeModel()
    model = GaussianNBModel()
    #model = KNeighborModel()
    #model = RForestModel()
    #model = SVCmodel()
    #model = LDAmodel()
    #model = LogRegModel()
    
    dfs = []
    
    for i in range(1,6):
        dfs.append(classify_glcm(model, file_loc))
        print(f'Epoch {i} done\n')
    
    save_excel(model, dfs, excel_loc)

    e = int(time.time() - start_time)
    print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
