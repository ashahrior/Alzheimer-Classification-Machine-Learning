from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import itertools

'''
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
'''
title = 'KNbr_'

parameters_list = [['uniform', 'distance'], ['auto', 'ball_tree', 'kd_tree', 'brute']]

combos = [list(x) for x in list(itertools.product(*[['uniform', 'distance'], ['auto', 'ball_tree', 'kd_tree', 'brute']]))]

headers = ['WEIGHTS','ALGORITHM','BEST_ACCURACY,','COMPONENT-NO.','%-ACCURACY']

def make_model(c, train_X, train_Y, test_X, test_Y):
    
    kneighbors_model = KNeighborsClassifier(
        weights=combos[c][0],algorithm=combos[c][1]
        )
    kneighbors_model.fit(train_X,train_Y)
    res_model = kneighbors_model.predict(test_X)

    return accuracy_score(test_Y, res_model)

