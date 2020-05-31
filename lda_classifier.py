import itertools
import os
import time
import xlsxwriter as xl

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import lda_classifier_combo
from functional_modules import pca_module

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


####### Main Program .... #####

def lin_disc_classifier_func():
    # Feature of GLCM is Loaded.
    all_GLCM_features = np.load(flocate.GLCM_all_case_feats_file, allow_pickle=True)

    # The file contains 302 rows and 778 columns. The Last column is the target value.
    # all_X contains the 778 columns of features where each row represent a Data volume
    # all_Y contains the target value from the last column.
    all_X = all_GLCM_features[:, :777]
    all_Y = all_GLCM_features[:, 777]

    # number_principal_components = 300

    # A list of all possible parameters and their values collected from sklearn site
    # Paremeters with their possible values for LogReg
    solver = ['svd', 'lsqr', 'eigen']
    shrinkage = ['auto', 0.1, 0.25, 0.5, 0.75, 0.99, None]
    store_covariance = [True, False]
    tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]

    parameters_list = [solver, shrinkage, store_covariance, tol]
    #combo_list = list(itertools.product(*parameters_list))
    #there are initially total 294 combos
    #after checking 210 combos are successful. model will work on that.
    '''
    n_comp = 300
    combos = 210
    total_iterations = 300*210 = 63000
    
    '''
    combo_list = lda_classifier_combo.lda_combos
    number_of_combos = len(combo_list)

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"
    
    '''st, fin = 0, 100
    serial = 1
    st, fin = 100, 200
    serial = 2'''
    st, fin = 200, 300
    serial = 3

    inner_iter = 0
    line = 0

    success = 0
    fail = 0
    #successful_combos = []

    outWorkbook = xl.Workbook(excel_loc+'lda_pca_{}.xlsx'.format(serial))
    outSheet1 = outWorkbook.add_worksheet()
    
    cols = ['A1','B1','C1','D1','E1','F1']
    heads = ['SOLVER','SHRINKAGE','STORE-COVARIANCE','TOLERANCE','BEST-ACCURACY','COMPONENT-NO.']

    for v in range(len(cols)):
        outSheet1.write(cols[v], heads[v])

    for compo in range(st, fin):
        all_p_comp = pca_module.applyPCA(all_X, compo+1)
        
        # Splitting Train, Test
        train_X, test_X, train_Y, test_Y = train_test_split(all_p_comp, all_Y, test_size=0.3)  
        
        slda = 0  # Linear Discriminant Analysis score
        for c in range(number_of_combos):
            try:
                slvr = combo_list[c][0]
                shrnk = combo_list[c][1]
                cov = combo_list[c][2]
                t = combo_list[c][3]
                inner_iter += 1
                print('Iteration #', inner_iter)

                lda_model = LinearDiscriminantAnalysis(
                    solver = slvr, shrinkage = shrnk,
                    priors=None, n_components=None,
                    store_covariance = cov, tol = t
                )
                lda_model.fit(train_X, train_Y)
                res_lda = lda_model.predict(test_X)
                score_lda = accuracy_score(test_Y, res_lda)

                print('With combo #{} , #{} Combinations Successful!'.format(c+1, success+1))
                success += 1
                print(
                    'LDA_Classifier >> Comp:', compo+1, ' solver:', slvr,
                    ' shrinkage:', shrnk,' store_covariance:', cov,' tolerance:',t,' Score:', score_lda, ' Highest Score:', slda
                )
                #successful_combos.append([slvr,shrnk,cov,t])
                if score_lda > slda:
                    print('New highest accuracy:', score_lda, '>', slda)
                    slda = score_lda
                    
                    outSheet1.write(line+1, 0, slvr)
                    outSheet1.write(line+1, 1, shrnk)
                    outSheet1.write(line+1, 2, cov)
                    outSheet1.write(line+1, 3, t)
                    outSheet1.write(line+1, 4, slda)
                    outSheet1.write(line+1, 5, compo+1)
                    print('At line {} writing file for component {}'.format(line+1,compo+1))
                    line+=1
                    time.sleep(2)
            except:
                #print('Combo failed at',c+1)
                #fail += 1
                pass
        time.sleep(2)
        os.system('cls')
    
    outWorkbook.close()
    print('All done!')
    '''print('Successful combos #{}'.format(len(successful_combos)))
    print('Counted successful combos %d'%success)
    print('Counted failed combos %d'%fail)
    print(successful_combos)'''

if __name__ == "__main__":
    lin_disc_classifier_func()
