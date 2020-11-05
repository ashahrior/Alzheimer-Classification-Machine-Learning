import itertools
import time, sys
import xlsxwriter as xl

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE


from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Classifier:
    def __init__(self, title: str, parameters_list: list, combos: list, headers: list):
        self.__title = title
        self.__parameters_list = parameters_list
        self.__combos = combos
        self.__headers = headers + ['COMPONENT-NO.', '%-ACCURACY', 'BEST_ACCURACY']

    def get_title(self):
        return self.__title

    def get_headers(self):
        return self.__headers

    def get_combos(self):
        return self.__combos

    def __get_accuracy(self, model, train_X, train_Y, test_X, test_Y):
        model.fit(train_X, train_Y)
        predictions = model.predict(test_X)
        accuracy = accuracy_score(test_Y, predictions)
        confusion = confusion_matrix(test_Y, predictions)
        report = classification_report(test_Y, predictions)
        return accuracy, confusion, report
        #return accuracy


class DTreeModel(Classifier):
    def __init__(self):
        self.__title = 'DTree_'
        self.__parameters_list = [['gini', 'entropy'], ['best', 'random'], [
            1.0, 'auto', 'sqrt', 'log2', None], ['balanced', None]]
        self.__headers = ['CRITERION', 'SPLITTER',
                          'MAX_FEATURES', 'CLASS_WEIGHT']
        #self.__combos = list(itertools.product(*self.parameters_list))
        self.__combos = [
            ['gini', 'best', 1.0, 'balanced'], ['gini', 'best', 1.0, None], ['gini', 'best', 'auto', 'balanced'], ['gini', 'best', 'auto', None], ['gini', 'best', 'sqrt', 'balanced'], ['gini', 'best', 'sqrt', None], ['gini', 'best', 'log2', 'balanced'], ['gini', 'best', 'log2', None], ['gini', 'best', None, 'balanced'], ['gini', 'best', None, None], ['gini', 'random', 1.0, 'balanced'], ['gini', 'random', 1.0, None], ['gini', 'random', 'auto', 'balanced'], ['gini', 'random', 'auto', None], ['gini', 'random', 'sqrt', 'balanced'], ['gini', 'random', 'sqrt', None], ['gini', 'random', 'log2', 'balanced'], ['gini', 'random', 'log2', None], ['gini', 'random', None, 'balanced'], ['gini', 'random', None, None], ['entropy', 'best', 1.0, 'balanced'], [
                'entropy', 'best', 1.0, None], ['entropy', 'best', 'auto', 'balanced'], ['entropy', 'best', 'auto', None], ['entropy', 'best', 'sqrt', 'balanced'], ['entropy', 'best', 'sqrt', None], ['entropy', 'best', 'log2', 'balanced'], ['entropy', 'best', 'log2', None], ['entropy', 'best', None, 'balanced'], ['entropy', 'best', None, None], ['entropy', 'random', 1.0, 'balanced'], ['entropy', 'random', 1.0, None], ['entropy', 'random', 'auto', 'balanced'], ['entropy', 'random', 'auto', None], ['entropy', 'random', 'sqrt', 'balanced'], ['entropy', 'random', 'sqrt', None], ['entropy', 'random', 'log2', 'balanced'], ['entropy', 'random', 'log2', None], ['entropy', 'random', None, 'balanced'], ['entropy', 'random', None, None]
        ]

        super(DTreeModel, self).__init__(self.__title,
                                         self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        #print('Applying combos >> ', self.__combos[c])
        self.__model = DecisionTreeClassifier(
            criterion=self.__combos[c][0], splitter=self.__combos[c][
                1], max_features=self.__combos[c][2], class_weight=self.__combos[c][3]
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
        self.__model = GaussianNB(var_smoothing=self.__combos[c][0])
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class KNeighborModel(Classifier):
    def __init__(self):
        self.__title = 'KNbr_'
        self.__parameters_list = [['uniform', 'distance'], ['auto']]
        self.__combos = [list(x) for x in list(
            itertools.product(*self.__parameters_list))]
        self.__headers = ['WEIGHTS', 'ALGORITHM']
        super(KNeighborModel, self).__init__(self.__title,
                                             self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = KNeighborsClassifier(weights=self.__combos[c][0], algorithm=self.__combos[c][1]
                                            )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class LDAmodel(Classifier):
    def __init__(self):
        self.__title = 'LDA_'
        solver = ['svd', 'lsqr']
        shrinkage = ['auto', 0.1, 0.25, 0.5, 0.75, 0.99, None]
        tol = [0.1, 0.01]
        self.__parameters_list = [solver, shrinkage, tol]
        #combo_list = list(itertools.product(*self.__parameters_list))
        # there are 28 combos.  16 succeeds.
        self.__headers = ['SOLVER', 'SHRINKAGE','TOLERANCE']
        self.__combos = [
            ['svd', None, 0.1], ['svd', None, 0.01], ['lsqr', 'auto', 0.1], ['lsqr', 'auto', 0.01], ['lsqr', 0.1, 0.1], ['lsqr', 0.1, 0.01], ['lsqr', 0.25, 0.1], ['lsqr', 0.25, 0.01], [
            'lsqr', 0.5, 0.1], ['lsqr', 0.5, 0.01], ['lsqr', 0.75, 0.1], ['lsqr', 0.75, 0.01], ['lsqr', 0.99, 0.1], ['lsqr', 0.99, 0.01], ['lsqr', None, 0.1], ['lsqr', None, 0.01]
        ]
        super(LDAmodel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = LinearDiscriminantAnalysis(
            solver=self.__combos[c][0], shrinkage=self.__combos[c][1], store_covariance=True, tol=self.__combos[c][2]
        )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class LogRegModel(Classifier):
    def __init__(self):
        self.__title = 'LogReg_'
        penalty = ['l1', 'l2', 'none']
        tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
        fit_intercept = [False, True]
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        multiclass = ['auto', 'ovr']
        self.__parameters_list = [penalty, tol, fit_intercept, solver, multiclass]
        #combo_list = list(itertools.product(*parameters_list))
        # There are 420 combos initially. But after checking 308 succeed
        self.__headers = ['PENALTY', 'TOLERANCE', 'FIT_INTERCEPT', 'SOLVER', 'MULTICLASS']
        self.__combos = [
            ['l1', 0.1, False, 'liblinear', 'auto'], ['l1', 0.1, False, 'liblinear', 'ovr'], ['l1', 0.1, False, 'saga', 'auto'], ['l1', 0.1, False, 'saga', 'ovr'], ['l1', 0.1, True, 'liblinear', 'auto'], ['l1', 0.1, True, 'liblinear', 'ovr'], ['l1', 0.1, True, 'saga', 'auto'], ['l1', 0.1, True, 'saga', 'ovr'], ['l1', 0.01, False, 'liblinear', 'auto'], ['l1', 0.01, False, 'liblinear', 'ovr'], ['l1', 0.01, False, 'saga', 'auto'], ['l1', 0.01, False, 'saga', 'ovr'], ['l1', 0.01, True, 'liblinear', 'auto'], ['l1', 0.01, True, 'liblinear', 'ovr'], ['l1', 0.01, True, 'saga', 'auto'], ['l1', 0.01, True, 'saga', 'ovr'], ['l1', 0.001, False, 'liblinear', 'auto'], ['l1', 0.001, False, 'liblinear', 'ovr'], ['l1', 0.001, False, 'saga', 'auto'], ['l1', 0.001, False, 'saga', 'ovr'], ['l1', 0.001, True, 'liblinear', 'auto'], ['l1', 0.001, True, 'liblinear', 'ovr'], ['l1', 0.001, True, 'saga', 'auto'], ['l1', 0.001, True, 'saga', 'ovr'], ['l1', 0.0001, False, 'liblinear', 'auto'], ['l1', 0.0001, False, 'liblinear', 'ovr'], ['l1', 0.0001, False, 'saga', 'auto'], ['l1', 0.0001, False, 'saga', 'ovr'], ['l1', 0.0001, True, 'liblinear', 'auto'], ['l1', 0.0001, True, 'liblinear', 'ovr'], ['l1', 0.0001, True, 'saga', 'auto'], ['l1', 0.0001, True, 'saga', 'ovr'], ['l1', 0.2, False, 'liblinear', 'auto'], ['l1', 0.2, False, 'liblinear', 'ovr'], ['l1', 0.2, False, 'saga', 'auto'], ['l1', 0.2, False, 'saga', 'ovr'], ['l1', 0.2, True, 'liblinear', 'auto'], ['l1', 0.2, True, 'liblinear', 'ovr'], ['l1', 0.2, True, 'saga', 'auto'], ['l1', 0.2, True, 'saga', 'ovr'], ['l1', 0.02, False, 'liblinear', 'auto'], ['l1', 0.02, False, 'liblinear', 'ovr'], ['l1', 0.02, False, 'saga', 'auto'], ['l1', 0.02, False, 'saga', 'ovr'], ['l1', 0.02, True, 'liblinear', 'auto'], ['l1', 0.02, True, 'liblinear', 'ovr'], ['l1', 0.02, True, 'saga', 'auto'], ['l1', 0.02, True, 'saga', 'ovr'], ['l1', 0.002, False, 'liblinear', 'auto'], ['l1', 0.002, False, 'liblinear', 'ovr'], ['l1', 0.002, False, 'saga', 'auto'], ['l1', 0.002, False, 'saga', 'ovr'], ['l1', 0.002, True, 'liblinear', 'auto'], ['l1', 0.002, True, 'liblinear', 'ovr'], ['l1', 0.002, True, 'saga', 'auto'], ['l1', 0.002, True, 'saga', 'ovr'], ['l2', 0.1, False, 'newton-cg', 'auto'], ['l2', 0.1, False, 'newton-cg', 'ovr'], ['l2', 0.1, False, 'lbfgs', 'auto'], ['l2', 0.1, False, 'lbfgs', 'ovr'], ['l2', 0.1, False, 'liblinear', 'auto'], ['l2', 0.1, False, 'liblinear', 'ovr'], ['l2', 0.1, False, 'sag', 'auto'], ['l2', 0.1, False, 'sag', 'ovr'], ['l2', 0.1, False, 'saga', 'auto'], ['l2', 0.1, False, 'saga', 'ovr'], ['l2', 0.1, True, 'newton-cg', 'auto'], ['l2', 0.1, True, 'newton-cg', 'ovr'], ['l2', 0.1, True, 'lbfgs', 'auto'], ['l2', 0.1, True, 'lbfgs', 'ovr'], ['l2', 0.1, True, 'liblinear', 'auto'], ['l2', 0.1, True, 'liblinear', 'ovr'], ['l2', 0.1, True, 'sag', 'auto'], ['l2', 0.1, True, 'sag', 'ovr'], ['l2', 0.1, True, 'saga', 'auto'], ['l2', 0.1, True, 'saga', 'ovr'], ['l2', 0.01, False, 'newton-cg', 'auto'], ['l2', 0.01, False, 'newton-cg', 'ovr'], ['l2', 0.01, False, 'lbfgs', 'auto'], ['l2', 0.01, False, 'lbfgs', 'ovr'], ['l2', 0.01, False, 'liblinear', 'auto'], ['l2', 0.01, False, 'liblinear', 'ovr'], ['l2', 0.01, False, 'sag', 'auto'], ['l2', 0.01, False, 'sag', 'ovr'], ['l2', 0.01, False, 'saga', 'auto'], ['l2', 0.01, False, 'saga', 'ovr'], ['l2', 0.01, True, 'newton-cg', 'auto'], ['l2', 0.01, True, 'newton-cg', 'ovr'], ['l2', 0.01, True, 'lbfgs', 'auto'], ['l2', 0.01, True, 'lbfgs', 'ovr'], ['l2', 0.01, True, 'liblinear', 'auto'], ['l2', 0.01, True, 'liblinear', 'ovr'], ['l2', 0.01, True, 'sag', 'auto'], ['l2', 0.01, True, 'sag', 'ovr'], ['l2', 0.01, True, 'saga', 'auto'], ['l2', 0.01, True, 'saga', 'ovr'], ['l2', 0.001, False, 'newton-cg', 'auto'], ['l2', 0.001, False, 'newton-cg', 'ovr'], ['l2', 0.001, False, 'lbfgs', 'auto'], ['l2', 0.001, False, 'lbfgs', 'ovr'], ['l2', 0.001, False, 'liblinear', 'auto'], ['l2', 0.001, False, 'liblinear', 'ovr'], ['l2', 0.001, False, 'sag', 'auto'], ['l2', 0.001, False, 'sag', 'ovr'], ['l2', 0.001, False, 'saga', 'auto'], ['l2', 0.001, False, 'saga', 'ovr'], ['l2', 0.001, True, 'newton-cg', 'auto'], ['l2', 0.001, True, 'newton-cg', 'ovr'], ['l2', 0.001, True, 'lbfgs', 'auto'], ['l2', 0.001, True, 'lbfgs', 'ovr'], ['l2', 0.001, True, 'liblinear', 'auto'], ['l2', 0.001, True, 'liblinear', 'ovr'], ['l2', 0.001, True, 'sag', 'auto'], ['l2', 0.001, True, 'sag', 'ovr'], ['l2', 0.001, True, 'saga', 'auto'], ['l2', 0.001, True, 'saga', 'ovr'], ['l2', 0.0001, False, 'newton-cg', 'auto'], ['l2', 0.0001, False, 'newton-cg', 'ovr'], ['l2', 0.0001, False, 'lbfgs', 'auto'], ['l2', 0.0001, False, 'lbfgs', 'ovr'], ['l2', 0.0001, False, 'liblinear', 'auto'], ['l2', 0.0001, False, 'liblinear', 'ovr'], ['l2', 0.0001, False, 'sag', 'auto'], ['l2', 0.0001, False, 'sag', 'ovr'], ['l2', 0.0001, False, 'saga', 'auto'], ['l2', 0.0001, False, 'saga', 'ovr'], ['l2', 0.0001, True, 'newton-cg', 'auto'], ['l2', 0.0001, True, 'newton-cg', 'ovr'], ['l2', 0.0001, True, 'lbfgs', 'auto'], ['l2', 0.0001, True, 'lbfgs', 'ovr'], ['l2', 0.0001, True, 'liblinear', 'auto'], ['l2', 0.0001, True, 'liblinear', 'ovr'], ['l2', 0.0001, True, 'sag', 'auto'], ['l2', 0.0001, True, 'sag', 'ovr'], ['l2', 0.0001, True, 'saga', 'auto'], ['l2', 0.0001, True, 'saga', 'ovr'], ['l2', 0.2, False, 'newton-cg', 'auto'], ['l2', 0.2, False, 'newton-cg', 'ovr'], ['l2', 0.2, False, 'lbfgs', 'auto'], ['l2', 0.2, False, 'lbfgs', 'ovr'], ['l2', 0.2, False, 'liblinear', 'auto'], ['l2', 0.2, False, 'liblinear', 'ovr'], ['l2', 0.2, False, 'sag', 'auto'], ['l2', 0.2, False, 'sag', 'ovr'], ['l2', 0.2, False, 'saga', 'auto'], ['l2', 0.2, False, 'saga', 'ovr'], ['l2', 0.2, True, 'newton-cg', 'auto'], ['l2', 0.2, True, 'newton-cg', 'ovr'], ['l2', 0.2, True, 'lbfgs', 'auto'], ['l2', 0.2, True, 'lbfgs', 'ovr'], ['l2', 0.2, True, 'liblinear', 'auto'], ['l2', 0.2, True, 'liblinear', 'ovr'], ['l2', 0.2, True, 'sag', 'auto'], ['l2', 0.2, True, 'sag', 'ovr'], ['l2', 0.2, True, 'saga', 'auto'], ['l2', 0.2, True, 'saga', 'ovr'], ['l2', 0.02, False, 'newton-cg', 'auto'], ['l2', 0.02, False, 'newton-cg', 'ovr'], ['l2', 0.02, False, 'lbfgs', 'auto'], ['l2', 0.02, False, 'lbfgs', 'ovr'], ['l2', 0.02, False, 'liblinear', 'auto'], ['l2', 0.02, False, 'liblinear', 'ovr'], ['l2', 0.02, False, 'sag', 'auto'], ['l2', 0.02, False, 'sag', 'ovr'], ['l2', 0.02, False, 'saga', 'auto'], ['l2', 0.02, False, 'saga', 'ovr'], ['l2', 0.02, True, 'newton-cg', 'auto'], ['l2', 0.02, True, 'newton-cg', 'ovr'], ['l2', 0.02, True, 'lbfgs', 'auto'], ['l2', 0.02, True, 'lbfgs', 'ovr'], ['l2', 0.02, True, 'liblinear', 'auto'], ['l2', 0.02, True, 'liblinear', 'ovr'], ['l2', 0.02, True, 'sag', 'auto'], ['l2', 0.02, True, 'sag', 'ovr'], ['l2', 0.02, True, 'saga', 'auto'], ['l2', 0.02, True, 'saga', 'ovr'], ['l2', 0.002, False, 'newton-cg', 'auto'], ['l2', 0.002, False, 'newton-cg', 'ovr'], ['l2', 0.002, False, 'lbfgs', 'auto'], ['l2', 0.002, False, 'lbfgs', 'ovr'], ['l2', 0.002, False, 'liblinear', 'auto'], ['l2', 0.002, False, 'liblinear', 'ovr'], ['l2', 0.002, False, 'sag', 'auto'], ['l2', 0.002, False, 'sag', 'ovr'], ['l2', 0.002, False, 'saga', 'auto'], ['l2', 0.002, False, 'saga', 'ovr'], ['l2', 0.002, True, 'newton-cg', 'auto'], ['l2', 0.002, True, 'newton-cg', 'ovr'], ['l2', 0.002, True, 'lbfgs', 'auto'], ['l2', 0.002, True, 'lbfgs', 'ovr'], ['l2', 0.002, True, 'liblinear', 'auto'], ['l2', 0.002, True, 'liblinear', 'ovr'], ['l2', 0.002, True, 'sag', 'auto'], ['l2', 0.002, True, 'sag', 'ovr'], ['l2', 0.002, True, 'saga', 'auto'], ['l2', 0.002, True, 'saga', 'ovr'], ['none', 0.1, False, 'newton-cg', 'auto'], ['none', 0.1, False, 'newton-cg', 'ovr'], ['none', 0.1, False, 'lbfgs', 'auto'], ['none', 0.1, False, 'lbfgs', 'ovr'], ['none', 0.1, False, 'sag', 'auto'], ['none', 0.1, False, 'sag', 'ovr'], ['none', 0.1, False, 'saga', 'auto'], ['none', 0.1, False, 'saga', 'ovr'], ['none', 0.1, True, 'newton-cg', 'auto'], ['none', 0.1, True, 'newton-cg', 'ovr'], ['none', 0.1, True, 'lbfgs', 'auto'], ['none', 0.1, True, 'lbfgs', 'ovr'], ['none', 0.1, True, 'sag', 'auto'], ['none', 0.1, True, 'sag', 'ovr'], ['none', 0.1, True, 'saga', 'auto'], ['none', 0.1, True, 'saga', 'ovr'], ['none', 0.01, False, 'newton-cg', 'auto'], ['none', 0.01, False, 'newton-cg', 'ovr'], ['none', 0.01, False, 'lbfgs', 'auto'], ['none', 0.01, False, 'lbfgs', 'ovr'], ['none', 0.01, False, 'sag', 'auto'], ['none', 0.01, False, 'sag', 'ovr'], ['none', 0.01, False, 'saga', 'auto'], ['none', 0.01, False, 'saga', 'ovr'], ['none', 0.01, True, 'newton-cg', 'auto'], ['none', 0.01, True, 'newton-cg', 'ovr'], ['none', 0.01, True, 'lbfgs', 'auto'], ['none', 0.01, True, 'lbfgs', 'ovr'], ['none', 0.01, True, 'sag', 'auto'], ['none', 0.01, True, 'sag', 'ovr'], ['none', 0.01, True, 'saga', 'auto'], ['none', 0.01, True, 'saga', 'ovr'], ['none', 0.001, False, 'newton-cg', 'auto'], ['none', 0.001, False, 'newton-cg', 'ovr'], ['none', 0.001, False, 'lbfgs', 'auto'], ['none', 0.001, False, 'lbfgs', 'ovr'], ['none', 0.001, False, 'sag', 'auto'], ['none', 0.001, False, 'sag', 'ovr'], ['none', 0.001, False, 'saga', 'auto'], ['none', 0.001, False, 'saga', 'ovr'], ['none', 0.001, True, 'newton-cg', 'auto'], ['none', 0.001, True, 'newton-cg', 'ovr'], ['none', 0.001, True, 'lbfgs', 'auto'], ['none', 0.001, True, 'lbfgs', 'ovr'], ['none', 0.001, True, 'sag', 'auto'], ['none', 0.001, True, 'sag', 'ovr'], ['none', 0.001, True, 'saga', 'auto'], ['none', 0.001, True, 'saga', 'ovr'], ['none', 0.0001, False, 'newton-cg', 'auto'], ['none', 0.0001, False, 'newton-cg', 'ovr'], ['none', 0.0001, False, 'lbfgs', 'auto'], ['none', 0.0001, False, 'lbfgs', 'ovr'], ['none', 0.0001, False, 'sag', 'auto'], ['none', 0.0001, False, 'sag', 'ovr'], ['none', 0.0001, False, 'saga', 'auto'], ['none', 0.0001, False, 'saga', 'ovr'], ['none', 0.0001, True, 'newton-cg', 'auto'], ['none', 0.0001, True, 'newton-cg', 'ovr'], ['none', 0.0001, True, 'lbfgs', 'auto'], ['none', 0.0001, True, 'lbfgs', 'ovr'], ['none', 0.0001, True, 'sag', 'auto'], ['none', 0.0001, True, 'sag', 'ovr'], ['none', 0.0001, True, 'saga', 'auto'], ['none', 0.0001, True, 'saga', 'ovr'], ['none', 0.2, False, 'newton-cg', 'auto'], ['none', 0.2, False, 'newton-cg', 'ovr'], ['none', 0.2, False, 'lbfgs', 'auto'], ['none', 0.2, False, 'lbfgs', 'ovr'], ['none', 0.2, False, 'sag', 'auto'], ['none', 0.2, False, 'sag', 'ovr'], ['none', 0.2, False, 'saga', 'auto'], ['none', 0.2, False, 'saga', 'ovr'], ['none', 0.2, True, 'newton-cg', 'auto'], ['none', 0.2, True, 'newton-cg', 'ovr'], ['none', 0.2, True, 'lbfgs', 'auto'], ['none', 0.2, True, 'lbfgs', 'ovr'], ['none', 0.2, True, 'sag', 'auto'], ['none', 0.2, True, 'sag', 'ovr'], ['none', 0.2, True, 'saga', 'auto'], ['none', 0.2, True, 'saga', 'ovr'], ['none', 0.02, False, 'newton-cg', 'auto'], ['none', 0.02, False, 'newton-cg', 'ovr'], ['none', 0.02, False, 'lbfgs', 'auto'], ['none', 0.02, False, 'lbfgs', 'ovr'], ['none', 0.02, False, 'sag', 'auto'], ['none', 0.02, False, 'sag', 'ovr'], ['none', 0.02, False, 'saga', 'auto'], ['none', 0.02, False, 'saga', 'ovr'], ['none', 0.02, True, 'newton-cg', 'auto'], ['none', 0.02, True, 'newton-cg', 'ovr'], ['none', 0.02, True, 'lbfgs', 'auto'], ['none', 0.02, True, 'lbfgs', 'ovr'], ['none', 0.02, True, 'sag', 'auto'], ['none', 0.02, True, 'sag', 'ovr'], ['none', 0.02, True, 'saga', 'auto'], ['none', 0.02, True, 'saga', 'ovr'], ['none', 0.002, False, 'newton-cg', 'auto'], ['none', 0.002, False, 'newton-cg', 'ovr'], ['none', 0.002, False, 'lbfgs', 'auto'], ['none', 0.002, False, 'lbfgs', 'ovr'], ['none', 0.002, False, 'sag', 'auto'], ['none', 0.002, False, 'sag', 'ovr'], ['none', 0.002, False, 'saga', 'auto'], ['none', 0.002, False, 'saga', 'ovr'], ['none', 0.002, True, 'newton-cg', 'auto'], ['none', 0.002, True, 'newton-cg', 'ovr'], ['none', 0.002, True, 'lbfgs', 'auto'], ['none', 0.002, True, 'lbfgs', 'ovr'], ['none', 0.002, True, 'sag', 'auto'], ['none', 0.002, True, 'sag', 'ovr'], ['none', 0.002, True, 'saga', 'auto'], ['none', 0.002, True, 'saga', 'ovr']
        ]
        
        super(LogRegModel, self).__init__(self.__title, self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = LogisticRegression(
            penalty=self.__combos[c][0], dual=False, tol=self.__combos[c][1],
            fit_intercept=self.__combos[c][2], max_iter=1000, random_state=0,
            solver=self.__combos[c][3], multi_class=self.__combos[c][4], warm_start=True, n_jobs=-2
        )
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)


class RForestModel(Classifier):
    def __init__(self):
        self.__title = 'RForest_'
        criterion = ['gini', 'entropy']
        max_features = ['auto', 'sqrt', 'log2']
        bootstrap = [True, False]
        oob_score = [True, False]
        warm_start = [True, False]
        self.__parameters_list = [ criterion, max_features, bootstrap, oob_score, warm_start]
        #self.__combos =[list(x) for x in list(itertools.product(*parameters_list))]
        # there are total 48 combinations originally
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
        self.__parameters_list = [
            ['linear', 'poly', 'rbf', 'sigmoid'], ['ovo']]
        self.__combos = [list(x) for x in list(
            itertools.product(*self.__parameters_list))]
        self.__headers = ['KERNEL', 'DECISION_FUNC_SHAPE', ]
        super(SVCmodel, self).__init__(self.__title,
                                       self.__parameters_list, self.__combos, self.__headers)

    def make_model(self, c, train_X, train_Y, test_X, test_Y):
        self.__model = SVC(
            kernel=self.__combos[c][0], decision_function_shape=self.__combos[c][1])
        return self._Classifier__get_accuracy(self.__model, train_X, train_Y, test_X, test_Y)



def prepare_data(data_path):
    all_data = np.load(data_path, allow_pickle=True)
    shape = all_data.shape
    print('Data shape >> ', shape)
    all_X = all_data[:, :-1]
    all_Y = all_data[:, -1]
    print('Data distribution complete.\n')
    return all_X, all_Y, shape[0]-1


def applyPCA(X, no_comp):
    pca = PCA(n_components=no_comp)
    pcomp = pca.fit_transform(X)
    return pcomp


def save_conf_mat(classifier, feature_type, accuracy, confusion, report):
    #global file
    file = open(filename, "w")
    file.write('Accuracy: ' + str(accuracy*100))
    file.write('\n\n')
    file.write('Confusion matrix:\n'+str(confusion))
    file.write('\n\n')
    file.write('Classification report:\n'+str(report))
    file.write('\n\n\n')
    #pkl = model_fol + f'{classifier}_{feature_type}_model.sav'
    #pickle.dump(model, open(pkl, 'wb'))
    return


def train_model(df, headers, classifier, X, Y, serial=4, doCompo=False):
    combo_list = classifier.get_combos()
    number_of_combos = len(combo_list)
    print(classifier.get_title(), end=' ')
    print('Processing compo #', serial, end=' ')

    x = MinMaxScaler().fit_transform(X)
    if doCompo:
      x = applyPCA(x, serial)
      print('PCA done #%d' % (serial))

    success = 0
    best_score = 0
    fail = 0

    #oversample = SMOTE()
    #x, Y = oversample.fit_resample(x, Y)

    train_X, test_X, train_Y, test_Y = train_test_split(x, Y, test_size=0.2)
    
    for c in range(number_of_combos):
        print(classifier.get_title(), ' - Entering combo #', c+1, end=' ')
        try:
            score_model, confusion, report = classifier.make_model(c, train_X, train_Y, test_X, test_Y)
            #score_model = classifier.make_model(c, train_X, train_Y, test_X, test_Y)
            print('Compo #{} - #{} Combos Successful!\nScore: {}'.format(serial, success+1, score_model))
            success += 1
            #successful_combos.append(combo_list[c])
            if score_model > best_score:
                print(classifier.get_title(),'- New highest accuracy:', score_model, '>', best_score)
                print(combo_list[c])
                best_score = score_model
                #_scores.append(best_score)
                parameters = combo_list[c] + [serial, float( "{:.2f}".format(best_score * 100)) , best_score]
                d = {}
                for h, p in zip(headers, parameters):
                    d[h] = p
                df = df.append(d, ignore_index=True)
                if best_score > 0.7:
                    global feature_type
                    #save_conf_mat(classifier.get_title(), feature_type, score_model, confusion, report)
        except:
            print(classifier.get_title(), ' - Compo #', serial, ' -> failed')
            fail += 1
        print(classifier.get_title(),' - Exiting compo #%d - combo #%d\n' % (serial, c+1))
    print(classifier.get_title(), f' - Compo {serial} - all done.')
    print(classifier.get_title(), ' - Total combinations: ', number_of_combos,
          'Total success: ', success, 'Total failure:', fail, '\n')
    
    '''
    print()
    print(len(successful_combos))
    print(successful_combos)
    '''
    return df


def classify_feats(model, path):
  X, Y, limit = prepare_data(path)
  headers = model.get_headers()
  df = pd.DataFrame(columns=headers)
  #limit = 2 #set limit to 2 when doing VLAD
  start, finish = 1,161
  for serial in range(start, finish):
      df = train_model(df, headers, model, X, Y, serial, False)
      print('Serial #', serial, 'done.\n\n')
  return df


def save_excel(model, dfs, excel_loc, feat, epoch=1):
    title = model.get_title()
    save_as = excel_loc + f"{feat}-{title}pca-{epoch}_var.xlsx"
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

    glcm_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_GLCM\clahe_glcm_54.npy"
    
    hog_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\Big_HOGs\CLAHE-HOG-MERGED.npy"

    #vlad_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\VLAD_16_feat2.npy"
    vlad_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\New folder\VLAD_16_feat.npy"

    hog_mrmr_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\Big_HOGs\mrmr_minmax_hog_160.npy"

    result_loc = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\excels\FINALS\\'

    model = GaussianNBModel()
    #model = DTreeModel()
    #model = KNeighborModel()
    #model = SVCmodel()
    #model = RForestModel()  # takes time
    #model = LDAmodel()     # takes time
    #model = LogRegModel()  # takes a lot of time 
    
    no_fl = 2

    #feature_type = 'glcm'
    feature_type = 'hog'
    #feature_type = 'vlad'
    
    file_path = ''
    
    if feature_type == 'glcm':
        file_path = glcm_path
    elif feature_type == 'hog':
        file_path = hog_path
    else: file_path = vlad_path
    #file_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\Big_HOGs\clustered_hog.npy"
    file_path = hog_mrmr_path

    filename = result_loc + f'{model.get_title()}_pca_{feature_type}_XXX{no_fl}.txt'
    #file = open(filename, "w")

    dfs = []
    
    for i in range(1,2):
        dfs.append(classify_feats(model, file_path))
        print(f'Epoch {i} done\n')

    #save_excel(model, dfs, result_loc, feature_type, no_fl)
    
    #file.close()

    e = int(time.time() - start_time)
    print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

    print()
    df = dfs[0].sort_values(by=['%-ACCURACY'], ascending=False)
    print(df.head(10))
    sys.stdout.write('\a')

# glcm 1-20
# hog 140-160
# vlad 140-160
