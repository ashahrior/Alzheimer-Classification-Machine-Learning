from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from classifier_modules import lda_classifier_combo
import
title = 'LDA_'

solver = ['svd', 'lsqr', 'eigen']
shrinkage = ['auto', 0.1, 0.25, 0.5, 0.75, 0.99, None]
store_covariance = [True, False]
tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]

parameters_list = [solver, shrinkage, store_covariance, tol]
#combo_list = list(itertools.product(*parameters_list))
# there are initially total 294 combos
# after checking 210 combos are successful. model will work on that.

combos = lda_classifier_combo.lda_combos
number_of_combos = len(combos)

headers = ['SOLVER', 'SHRINKAGE', 'STORE-COVARIANCE',
           'TOLERANCE', 'COMPONENT-NO.', '%-ACCURACY', 'BEST-ACCURACY']


def make_model(c, train_X, train_Y, test_X, test_Y):
    lda_model = LinearDiscriminantAnalysis(
        solver=combos[c][0], shrinkage=combos[c][1], store_covariance=combos[c][2], tol=combos[c][3]
    )
    lda_model.fit(train_X, train_Y)
    res_model = lda_model.predict(test_X)

    return accuracy_score(test_Y, res_model)
