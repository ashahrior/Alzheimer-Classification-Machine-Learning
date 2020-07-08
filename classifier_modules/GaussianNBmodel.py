from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

title = 'GaussNB_'

var_smoothing = 0.000000001

parameters_list = [var_smoothing]

combos = [[var_smoothing]]

headers = ['VAR_SMOOTHING','COMPONENT-NO.','%-ACCURACY','BEST_ACCURACY,']

def make_model(c, train_X, train_Y, test_X, test_Y):
    gaussnb_model = GaussianNB(var_smoothing=combos[c][0])
    gaussnb_model.fit(train_X, train_Y)
    res_model = gaussnb_model.predict(test_X)
    
    return accuracy_score(test_Y, res_model)
