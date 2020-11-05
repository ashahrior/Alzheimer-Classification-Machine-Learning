import itertools
import time
import xlsxwriter as xl

import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


combinations = {

    #dtree - CRITERION	SPLITTER	MAX-FEATURES	CLASS-WEIGHT

    'DTree_': [
        ['gini', 'random', 1, None],
        ['gini', 'random', 'auto', 'balanced'],

    ],


    #gauss - VAR-SMOOTHING

    'GaussNB_': [
        [0.000000001],
        [0.000000001],
    ],


    #knbr - WEIGHTS	ALGORITHM

    'KNbr_': [
        ['distance', 'auto'],
        ['distance', 'auto'],
    ],


    #lda - SOLVER	SHRINKAGE	STORE-COVARIANCE	TOLERANCE

    'LDA_': [
        ['lsqr' , 0.1, True, 0.1],
        ['lsqr' , .25, None, 0.1],
    ],


    #logreg - PENALTY	DUAL	TOLERANCE	FIT-INTERCEPT	SOLVER	MULTICLASS	WARM-START

    'LogReg_': [
        
        ['l2', False, 0.1, False, 'newton-cg', 'auto', True],
        ['l2', False, 0.1, False, 'newton-cg', 'ovr', True],
    ],


    #rforest - CRITERION	MAX-FEATS	BOOTSTRAP	OOB-SCORE	WARM-START

    'RForest_': [

        ['entropy', 'sqrt', True, False, True],
        ['entropy', 'log2', True, True, True],

    ],


    #svc - KERNEL	DECISION-FUNC-SHAPE

    'SVC_': [
        ['linear', 'ovo'],
        ['linear', 'ovo'],
    ]
}

components = {
    #'DTree_', 'GaussNB_', 'KNbr_', 'LDA_', 'LogReg_', 'RForest_', 'SVC_'
    'DTree_': [6, 20], 
    'GaussNB_': [30, 37],
    'KNbr_': [16, 20],  
    'LDA_': [98, 55],  
    'LogReg_': [131, 113],
    'RForest_': [19, 63], 
    'SVC_': [75, 111]
}

model_fol = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\Models\\"

glcm_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_GLCM\clahe_glcm_54.npy"

hog_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\CLAHE-HOG-MERGED.npy"

vlad_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\VLAD_16_feat2.npy"

feature_path = {
    'glcm': glcm_path,
    'hog': hog_path,
    'vlad': vlad_path
}


def get_DTree(feature_type):
    combo = []
    if feature_type == 'glcm':
        combo = ['gini', 'random', 1, None]
    elif feature_type == 'hog':
        combo = ['gini', 'random', 'auto', 'balanced']
    return DecisionTreeClassifier(criterion=combo[0], splitter=combo[1], max_features=combo[2], class_weight=combo[3])


def get_GaussNB(feature_type):
    combo = [0.000000001]
    return GaussianNB(var_smoothing=combo[0])


def get_KNbr(feature_type):
    combo = []
    if feature_type == 'glcm':
        combo = ['distance', 'auto']
    elif feature_type == 'hog':
        combo = ['distance', 'auto']

    return KNeighborsClassifier(weights=combo[0], algorithm=combo[1])


def get_LDA(feature_type):
    combo = []
    if feature_type == 'glcm':
        combo = ['lsqr', 0.1, True, 0.1]
    elif feature_type == 'hog':
        combo = ['lsqr', .25, None, 0.1]

    return LinearDiscriminantAnalysis(solver=combo[0], shrinkage=combo[1], store_covariance=combo[2], tol=combo[3])


def get_LogReg(feature_type):
    combo = []
    if feature_type == 'glcm':
        combo = ['l2', False, 0.1, False, 'newton-cg', 'auto', True]
    elif feature_type == 'hog':
        combo = ['l2', False, 0.1, False, 'newton-cg', 'ovr', True]

    return LogisticRegression(penalty=combo[0], dual=combo[1], tol=combo[2], fit_intercept=combo[3], max_iter=1000, random_state=0, solver=combo[4], multi_class=combo[5], warm_start=combo[6])


def get_RForest(feature_type):
    combo = []
    if feature_type == 'glcm':
        combo = ['entropy', 'sqrt', True, False, True]
    elif feature_type == 'hog':
        combo = ['entropy', 'log2', True, True, True]

    return RandomForestClassifier(criterion=combo[0], max_features=combo[1], bootstrap=combo[2], oob_score=combo[3], warm_start=combo[4])


def get_SVC(feature_type):
    combo = []
    if feature_type == 'glcm':
        combo = ['linear', 'ovo']
    elif feature_type == 'hog':
        combo = ['linear', 'ovo']

    return SVC(kernel=combo[0], decision_function_shape=combo[1])


def get_index(feature_type):
    if feature_type == 'glcm':
        return 0
    elif feature_type == 'hog':
        return 1
    else: return 0


def get_classifier_properties(feature_type, classifier, index):
    global combinations
    global components
    model = ''

    if classifier == 'DTree_':
        model = get_DTree(feature_type)
    elif classifier == 'GaussNB_':
        model = get_GaussNB(feature_type)
    elif classifier == 'KNbr_':
        model = get_KNbr(feature_type)
    elif classifier == 'LDA_':
        model = get_LDA(feature_type)
    elif classifier == 'LogReg_':
        model = get_LogReg(feature_type)
    elif classifier == 'RForest_':
        model = get_RForest(feature_type)
    elif classifier == 'SVC_':
        model = get_SVC(feature_type)

    return model, combinations[classifier][index], components[classifier][index]


def applyPCA(feature, no_comp):
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components=no_comp)
    pcomp = pca.fit_transform(X)

    return pcomp


def prepare_data(data_path):
    all_data = np.load(data_path, allow_pickle=True)
    shape = all_data.shape
    print('Data shape >> ', shape)
    all_X = all_data[:, :-1]
    all_Y = all_data[:, -1]
    print('Data distribution complete.')

    return all_X, all_Y, shape[0]-1


def do_classification(X, Y, model, compo):
    x = applyPCA(X, compo)

    train_X, test_X, train_Y, test_Y = train_test_split(x, Y, test_size=0.2)

    model.fit(train_X, train_Y)

    predictions = model.predict(test_X)
    accuracy = accuracy_score(test_Y, predictions)
    confusion = confusion_matrix(test_Y, predictions)
    report = classification_report(test_Y, predictions)

    return accuracy, confusion, report


def save_credentials(file, classifier, feature_type, compo, accuracy, confusion, report):   
    file.write('Component: '+str(compo))
    file.write('\nAccuracy: ' + str(accuracy*100))
    file.write('\n\n')
    file.write('Confusion matrix:\n'+str(confusion))
    file.write('\n\n')
    file.write('Classification report:\n'+str(report))
    file.write('\n\n\n')
    #pkl = model_fol + f'{classifier}_{feature_type}_model.sav'

    #pickle.dump(model, open(pkl, 'wb'))
    print('Saved.')
    return


if __name__ == "__main__":
    
    # 'DTree_','GaussNB_','KNbr_','LDA_','LogReg_','RForest_','SVC_'
    classifier = 'GaussNB_'
    feature_type = 'glcm'

    index = get_index(feature_type)
    model, combo, compo = get_classifier_properties(feature_type, classifier, index)

    X, Y, limit = prepare_data(feature_path[feature_type])
    p = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\\train-test\\"
    file = open(p+f'{classifier}{feature_type}.txt', 'w+')
    #for compo in range(1, 161):
    for compo in [compo]:
        accuracy, confusion, report = do_classification(X, Y, model, compo)
        print('\nComponent #',compo)
        print('\nAccuracy: ', accuracy*100,'%')
        print('\nConfusion matrix:\n', confusion)
        print('\nReport:\n', report)
        if accuracy*100 > 80:
            save_credentials(classifier, feature_type, accuracy, confusion, report)
    file.close()
    print()
