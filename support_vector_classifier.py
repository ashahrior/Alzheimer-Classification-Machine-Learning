import itertools
import os
import time
import xlsxwriter as xl

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import pca_module

from sklearn.svm import SVC

####### Main Program .... #####


def sv_classifier_func():
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
    
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    decision_function_shape = ['ovo', 'ovr']
    
    param = [ kernel, decision_function_shape]

    feature_combo = list(itertools.product(*param))
    feature_combo = [list(x) for x in feature_combo]
    number_of_combos = len(feature_combo)
    '''
    st, fin = 0, 100  # 300
    serial = 1
    
    st, fin = 100, 200
    serial = 2 '''
    '''
    '''
    st, fin = 0, 300
    serial = 1
    

    success = 0
    fail = 0

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"

    inner_iter = 0
    line = 0

    #success_list = []
    
    outWorkbook = xl.Workbook(excel_loc+'svc_glcm_pca_2.xlsx')
    outSheet = outWorkbook.add_worksheet()

    outSheet.write('A1', 'KERNEL')
    outSheet.write('B1', 'DECISION-FUNC-SHAPE')
    outSheet.write('C1', 'BEST-ACCURACY')
    outSheet.write('D1', 'COMPONENT-NO.')
    outSheet.write('J1', 'ROW.')  # 9
    outSheet.write('K1', 'HIGHEST-ACCURACY')
    outSheet.write('L1', 'COMPONENT-NO.')
    
    highest_accuracy = 0
    final_row = 0
    cc = 0

    for compo in range(st, fin):
        all_p_comp = pca_module.applyPCA(all_X, compo+1)

        train_X, test_X, train_Y, test_Y = train_test_split(all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test
        ssvm = 0  # Support Vector Machine

        for c in range(number_of_combos):
            try:
                K = feature_combo[c][0]
                D = feature_combo[c][1]
                inner_iter += 1
                svm_model = SVC(kernel=K, decision_function_shape=D)
                svm_model.fit(train_X,train_Y)
                res_svm = svm_model.predict(test_X)
                score_svm = accuracy_score(test_Y,res_svm)
                #print('Score for',D,':',score_svm)
                #print('With combo #{} , #{} Combinations Successful!'.format(c+1, success+1))
                success += 1
                '''
                print(
                    '~ SVM_Classifier >> Comp:', compo +
                    1, '\nkernel: ', K, '   ---  decision_func_shape:', D, '\nScore:', score_svm, '  +++ Highest Score:', ssvm
                )'''

                if score_svm > ssvm:
                    print('New highest accuracy:', score_svm, '>', ssvm,' with - ',D)
                    ssvm = score_svm
                    
                    outSheet.write(line+1, 0, K)
                    outSheet.write(line+1, 1, D)
                    outSheet.write(line+1, 2, ssvm)
                    outSheet.write(line+1, 3, compo+1)
                    
                    #print('+++ +++ At line {} writing file for component {}'.format(line+1, compo+1))
                    line += 1
                    
            except:
                print('* Combo failed at', c+1)
                print()
                fail += 1
        if ssvm > highest_accuracy:
            highest_accuracy = ssvm
            cc = compo+1
            final_row = line+1
            print('### ### New highest', highest_accuracy, ' @', cc)
        print('component %d done.'%(compo+1))
        print()
    
    outSheet.write(1, 9, final_row)
    outSheet.write(1, 10, highest_accuracy)
    outSheet.write(1, 11, cc)
    #print('Writing highest accuracy {} for component #{}'.format(highest_accuracy, cc))
    #print()

    outWorkbook.close()
    print('All done!')
    print('Counted successful combos %d' % success)
    print('Counted failed combos %d' % fail)


if __name__ == "__main__":
    sv_classifier_func()
