from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
criterion = ['gini', 'entropy']
max_features = ['auto', 'sqrt', 'log2']
bootstrap = [True, False]
oob_score = [True, False]
warm_start = [True, False]
parameters_list = [ criterion, max_features, bootstrap, oob_score, warm_start]
feature_combo = list(itertools.product(*parameters_list))
# there are total 48 combinations originally
feature_combo = [list(x) for x in feature_combo]
'''

title = 'RForest_'

combos = [
    ['gini', 'auto', True, True, True], ['gini', 'auto', True, True, False], ['gini', 'auto', True, False, True], ['gini', 'auto', True, False, False], ['gini', 'auto', False, False, True], ['gini', 'auto', False, False, False], ['gini', 'sqrt', True, True, True], ['gini', 'sqrt', True, True, False], ['gini', 'sqrt', True, False, True], ['gini', 'sqrt', True, False, False], ['gini', 'sqrt', False, False, True], ['gini', 'sqrt', False, False, False], ['gini', 'log2', True, True, True], ['gini', 'log2', True, True, False], ['gini', 'log2', True, False, True], ['gini', 'log2', True, False, False], ['gini', 'log2', False, False, True], ['gini', 'log2', False, False, False], ['entropy', 'auto', True, True, True], ['entropy', 'auto', True, True, False], ['entropy', 'auto', True, False, True], ['entropy', 'auto', True, False, False], ['entropy', 'auto', False, False, True], ['entropy', 'auto', False, False, False], ['entropy', 'sqrt', True, True, True], ['entropy', 'sqrt', True, True, False], ['entropy', 'sqrt', True, False, True], ['entropy', 'sqrt', True, False, False], ['entropy', 'sqrt', False, False, True], ['entropy', 'sqrt', False, False, False], ['entropy', 'log2', True, True, True], ['entropy', 'log2', True, True, False], ['entropy', 'log2', True, False, True], ['entropy', 'log2', True, False, False], ['entropy', 'log2', False, False, True], ['entropy', 'log2', False, False, False]
]


headers = ['CRITERION', 'MAX_FEATURES', 'BOOTSTRAP',
           'OOB_SCORE', 'WARM_START', 'BEST_ACCURACY,', 'COMPONENT_NO.','%-ACCURACY']


def make_model(c, train_X, train_Y, test_X, test_Y):
    ran_forest_model = RandomForestClassifier(
        criterion=combos[c][0], max_features=combos[c][1], bootstrap=combos[c][2], oob_score=combos[c][3], warm_start=combos[c][4])

    ran_forest_model.fit(train_X, train_Y)
    res_model = ran_forest_model.predict(test_X)

    return accuracy_score(test_Y, res_model)
