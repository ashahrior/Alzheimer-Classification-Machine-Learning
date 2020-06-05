from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import itertools

'''
kernel=['linear', 'poly', 'rbf', 'sigmoid']
decision_function_shape = ['ovo', 'ovr']
'''
parameters_list = [['linear', 'poly', 'rbf', 'sigmoid'], ['ovo', 'ovr']]

title = 'SVC_'

combos = [list(x) for x in list(itertools.product(
    *[['linear', 'poly', 'rbf', 'sigmoid'], ['ovo', 'ovr']]))]

headers = ['KERNEL', 'DECISION_FUNC_SHAPE', 'BEST_ACCURACY,', 'COMPONENT_NO.','%-ACCURACY']


def make_model(c, train_X, train_Y, test_X, test_Y):
    svm_model = SVC(kernel=combos[c][0], decision_function_shape=combos[c][1])
    svm_model.fit(train_X, train_Y)
    res_model = svm_model.predict(test_X)

    return accuracy_score(test_Y, res_model)
