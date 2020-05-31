import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

# Applying PCA method. Nothing special here. it returns the Componenets of the features.


def applyPCA(feature, no_comp):
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components = no_comp)
    pcomp = pca.fit_transform(X)

    return pcomp


####### Main Program .... #####

# Feature of GLCM is Loaded.
all_feature = np.load('Feature/feature.npy', allow_pickle=True)

# The file contains 302 rows and 778 columns. The Last column is the target value.
# all_X contains the 778 columns of features where each row represent a Data volume
# all_Y contains the target value from the last column.
all_X = all_feature[:, :777]
all_Y = all_feature[:, 777]

# Some variables to hold the score ...

skn = 0  # KNeighbours


srf = 0  # Random Forest
ssvm = 0  # Support Vector Machine

# Repeat for different number of Components .
for i in range(300):  # For the number of principal Component
    # Returns the features with (i+1) components
    all_p_comp = applyPCA(all_X, i+1)

    train_X, test_X, train_Y, test_Y = train_test_split(
        all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test

    # Here is the list of all possible parameter and their values collected from Internet
    # A parameter and possible values for LogReg
    penalty = ['l1', 'l2', 'elasticnet', 'none']
    dual = [False, True]
    tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
    fit_intercept = [False, True]
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    multiclass = ['auto', 'ovr', 'multinomial']
    warmStart = [True, False]

    # Now running loops for each of the parameter . quite Simple
    for p in range(len(penalty)):    # For Penalty
        for d in range(len(dual)):      # For dual
            for t in range(len(tol)):       # For tolerance
                for fi in range(len(fit_intercept)):  # For fit_Intercept
                    for s in range(len(solver)):
                        for m in range(len(multiclass)):
                            for w in range(len(warmStart)):
                                try:  # Just a Try Case because All Combination is not possible
                                    # Creating the model with the parameters from the array
                                    log_reg_model = LogisticRegression(penalty=penalty[p], dual=dual[d],
                                                                       tol=tol[t], fit_intercept=fit_intercept[fi],
                                                                       max_iter=100, random_state=0, solver=solver[s],
                                                                       multi_class=multiclass[m], warm_start=warmStart[w])

                                    # Fitting The Data
                                    log_reg_model.fit(train_X, train_Y)
                                    res_log_reg = log_reg_model.predict(
                                        test_X)  # predicting
                                    score_log_reg = accuracy_score(
                                        test_Y, res_log_reg)  # Accuracy Calculating

                                    print('Log_Reg: Comp:', i+1, ' penalty:', penalty[p], ' Dual:', dual[d],
                                          ' Tol:', tol[t], ' FI:', fit_intercept[fi],
                                          ' solver:', solver[s], ' mulclass:', multiclass[m],
                                          ' warmStart:', warmStart[w],' Score:', score_log_reg)  # Just Printing The information

                                    # Checking whether the accuracy is highest or not.
                                    if score_log_reg > slg:
                                        print('Log_Reg: Comp:', i+1, ' penalty:', penalty[p], ' Dual:', dual[d],
                                              ' Tol:', tol[t], ' FI:', fit_intercept[fi],
                                              ' solver:', solver[s], ' mulclass:', multiclass[m],
                                              ' warmStart:', warmStart[w], ' Score:', score_log_reg)
                                        slg = score_log_reg
                                        # Printing The best param and Scores
                                        wait = input('Wait for me : ')
                                        # and Waiting for an enter for the next best Model

                                except:
                                    # The combination is not possible
                                    print('Could not create  the model')

    '''
    This section is the basic Implementation of the Models. I have just written
    the tunning code for the first model.For Simplicity the rest are commented. 
    '''

    ''' 
    kneighbors_model = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='auto',p=2)
    kneighbors_model.fit(train_X,train_Y)
    res_kneighbors = kneighbors_model.predict(test_X)
    score_kneighbors = accuracy_score(test_Y,res_kneighbors)
    if score_kneighbors > skn:
        print('KN : ',i+1,' ',score_kneighbors)
        skn = score_kneighbors
        wait = input('Wait for me : ')


    ran_forest_model = RandomForestClassifier(n_jobs=2,n_estimators=100,random_state=0)
    ran_forest_model.fit(train_X,train_Y)
    res_ran_forest = ran_forest_model.predict(test_X)
    score_ran_forest = accuracy_score(test_Y,res_ran_forest)
    if score_ran_forest > srf:
        print('RanFor : ',i+1,' ',score_ran_forest)
        srf = score_ran_forest
        wait = input('Wait for me : ')


    svm_model = SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo')
    svm_model.fit(train_X,train_Y)
    res_svm = svm_model.predict(test_X)
    score_svm = accuracy_score(test_Y,res_svm)
    if score_svm > ssvm:
        print('SVM : ',i+1,' ',score_svm)
        ssvm = score_svm
        wait = input('Wait for me : ')

    '''
