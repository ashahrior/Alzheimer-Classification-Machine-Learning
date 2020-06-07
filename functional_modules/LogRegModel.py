from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from functional_modules import log_reg_classifier_combos as log_reg_combo

title = 'LogReg_'

penalty = ['l1', 'l2', 'elasticnet', 'none']
dual = [False, True]
tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
fit_intercept = [False, True]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
multiclass = ['auto', 'ovr', 'multinomial']
warm = [True, False]

parameters_list = [penalty, dual, tol, fit_intercept, solver, multiclass, warm]
#combo_list = list(itertools.product(*parameters_list))
# after checking 924 combos were successful. model will work on that.

# now this has 336 combos after some more filtering
combos = log_reg_combo.logistic_regression_combos
number_of_combos = len(combos)

headers = ['PENALTY', 'DUAL', 'TOLERANCE', 'FIT_INTERCEPT', 'SOLVER', 'MULTICLASS', 'WARM_START', 'BEST_ACCURACY,', 'COMPONENT_NO.','%-ACCURACY']


def make_model(c, train_X, train_Y, test_X, test_Y):
    log_reg_model = LogisticRegression(
        penalty=combos[c][0], dual=combos[c][1], tol=combos[c][2],fit_intercept=combos[c][3], max_iter=100,
        random_state=0, solver=combos[c][4],
        multi_class=combos[c][5], warm_start= combos[c][6]
    )
    log_reg_model.fit(train_X, train_Y)
    res_model = log_reg_model.predict(test_X)

    return accuracy_score(test_Y, res_model)


#>>> multiclass = ['auto', 'ovr']
#>>> penalty = ['l1', 'l2', 'none']
#>>> parameters_list = [penalty, dual, tol, fit_intercept, solver, multiclass]
#>>> combo_list = list(itertools.product(*parameters_list))
#>>> len(combo_list)
#840
'''
penalty = ['l1', 'l2', 'none']
dual = [False, True]
tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
fit_intercept = [False, True]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
multiclass = ['auto', 'ovr']
warm = [True, False]

parameters_list = [penalty, dual, tol, fit_intercept, solver, multiclass, warm]
combo_list = list(itertools.product(*parameters_list))
'''
