import pickle

import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

combinations = {

    #dtree - CRITERION	SPLITTER	MAX-FEATURES	CLASS-WEIGHT

    'DTree_': [
        
        ['entropy', 'random', 'sqrt', 'balanced'],
        ['entropy', 'random', 1, None],
        ['gini', 'random', None, None],
        ['gini', 'best', 'auto', 'balanced'],
        ['entropy', 'random', 1, 'balanced']
    ],


    #gauss - VAR-SMOOTHING

    'GaussNB_': [
        [0.000000001]
    ],


    #knbr - WEIGHTS	ALGORITHM

    'KNbr_': [
        ['distance', 'auto']
    ],


    #lda - SOLVER	SHRINKAGE	STORE-COVARIANCE	TOLERANCE

    'LDA_': [
        ['lsqr' , 0.1, True, 0.1],
        ['lsqr' , 'auto', True, 0.1],
        ['svd' , None, True, 0.1]
    ],


    #logreg - PENALTY	DUAL	TOLERANCE	FIT-INTERCEPT	SOLVER	MULTICLASS	WARM-START

    'LogReg_': [
        
        ['none', False, 0.1, True, 'newton-cg', 'auto', True],
        ['none', False, 0.01, True, 'newton-cg', 'ovr', True],
        ['none', False, 0.01, False, 'newton-cg',	'auto', True],
        ['none', False, 0.1, False, 'lbfgs', 'ovr', True],
        ['l2', False, 0.1, False, 'newton-cg', 'auto', True],
    ],


    #rforest - CRITERION	MAX-FEATS	BOOTSTRAP	OOB-SCORE	WARM-START

    'RForest_': [

        ['entropy', 'auto', False, False, False],
        ['entropy', 'sqrt', True, True, False],
        ['gini', 'auto', False, False, True],
        ['gini', 'log2', True, True, True],
        ['gini', 'auto', True, True, False]
    ],


    #svc - KERNEL	DECISION-FUNC-SHAPE

    'SVC_': [
        ['linear', 'ovo'],
        ['rbf', 'ovo']
    ]
}

compos = {
    #'DTree_', 'GaussNB_', 'KNbr_', 'LDA_', 'LogReg_', 'RForest_', 'SVC_'
    'DTree_': [7, 27, 16, 32, 3],  # [30, 8, 18, 22, 26],
    'GaussNB_': [24],  # [53],
    'KNbr_': [72],  # [239],
    'LDA_': [100, 19, 49],  # [261, 120, 38],
    'LogReg_': [95, 53, 62, 110, 141],  # [127, 218, 209, 263, 277],
    'RForest_': [12, 13, 14, 24, 20],  # [41, 39, 63, 36, 62],
    'SVC_': [127, 112]  # [81]
}

model_fol = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\Models\\"

feature_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\all_clean_glcm_54.npy"


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


if __name__ == "__main__":
    X, Y, limit = prepare_data(feature_path)
    print('Data prepared')

    # 'DTree_','GaussNB_','KNbr_','LDA_','LogReg_','RForest_','SVC_'

    classifier = 'LogReg_'
    combos = combinations[classifier]
    components = compos[classifier]

    for c in range(len(components)):
        x = applyPCA(X, components[c])
        print(f'comp-{components[c]} applied')

        models = {
            'DTree_' : DecisionTreeClassifier(criterion=combos[c][0], splitter=combos[c][1], max_features=combos[c][2], class_weight=combos[c][3]) ,

            'GaussNB_': GaussianNB(var_smoothing=combos[c][0]),
            
            'KNbr_': KNeighborsClassifier(weights=combos[c][0], algorithm=combos[c][1]),
            
            'LDA_': LinearDiscriminantAnalysis(solver=combos[c][0], shrinkage=combos[c][1], store_covariance=combos[c][2], tol=combos[c][3]),
            
            'LogReg_': LogisticRegression(penalty=combos[c][0], dual=combos[c][1], tol=combos[c][2], fit_intercept=combos[c][3], max_iter=600, random_state=0, solver=combos[c][4], multi_class=combos[c][5], warm_start=combos[c][6]) , 

            'RForest_': RandomForestClassifier(criterion=combos[c][0], max_features=combos[c][1], bootstrap=combos[c][2], oob_score=combos[c][3], warm_start=combos[c][4]),
            
            'SVC_': SVC(kernel=combos[c][0], decision_function_shape=combos[c][1])

        }

        train_X, test_X, train_Y, test_Y = train_test_split(x, Y, test_size=0.3)
        print(f'train test split done #{c+1}')

        p = model_fol + r"train-test\\"

        f = p + f'{classifier}_all-clean_glcm_combo-{c+1}_compo-{components[c]}_'
        np.save(f + 'test_X', test_X)
        np.save(f + 'test_Y', test_Y)
        print(f'Train-test saved #{c+1}')

        model = models[classifier]

        model.fit(train_X, train_Y)
        print(f'model fit complete #{c+1}')

        pkl = model_fol + f'{classifier}_all-clean_glcm_combo-{c+1}_compo-{components[c]}_model.sav'
        
        pickle.dump(model, open(pkl, 'wb'))
        
        print(f'{pkl} saved #{c+1}')
        print()
