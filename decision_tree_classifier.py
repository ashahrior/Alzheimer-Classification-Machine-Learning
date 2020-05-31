import itertools
import os
import time
import xlsxwriter as xl

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import dtree_classifier_combo
from functional_modules import pca_module

from sklearn.tree import DecisionTreeClassifier

####### Main Program .... #####


def decision_tree_classifier_func():
    # Feature of GLCM is Loaded.
    all_GLCM_features = np.load(
        flocate.GLCM_all_case_feats_file, allow_pickle=True)

    # The file contains 302 rows and 778 columns. The Last column is the target value.
    # all_X contains the 778 columns of features where each row represent a Data volume
    # all_Y contains the target value from the last column.
    all_X = all_GLCM_features[:, :777]
    all_Y = all_GLCM_features[:, 777]

    # number_principal_components = 300

    # A list of all possible parameters and their values collected from sklearn site
    # Paremeters with their possible values for LogReg
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_features = [1, 1.0, 'auto', 'sqrt', 'log2', None]
    class_weight = ['balanced', None]

    parameters_list = [criterion, splitter, max_features, class_weight]
    #combo_list = list(itertools.product(*parameters_list))
    # there are total 48 combinations originally
    '''
    n_comp = 300
    combos = 48
    total_iterations = 300*48 = 14400
    '''
    combo_list = dtree_classifier_combo.dtree_combos

    number_of_combos = len(combo_list)

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"

    inner_iter = 0
    line = 0

    st, fin = 0, 300
    success = 0
    '''
    fail = 0
    successful_combos = []
    '''

    outWorkbook = xl.Workbook(excel_loc+'decTree_glcm_pca.xlsx')
    outSheet = outWorkbook.add_worksheet()

    outSheet.write('A1', 'CRITERION')
    outSheet.write('B1', 'SPLITTER')
    outSheet.write('C1', 'MAX-FEATURES')
    outSheet.write('D1', 'CLASS-WEIGHT')
    outSheet.write('E1', 'BEST-ACCURACY')
    outSheet.write('F1', 'COMPONENT-NO.')

    for compo in range(st, fin):
        all_p_comp = pca_module.applyPCA(all_X, compo+1)

        train_X, test_X, train_Y, test_Y = train_test_split(
            all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test
        sdt = 0  # Decision Tree classifier score

        for c in range(number_of_combos):
            try:
                crt = combo_list[c][0]
                sp = combo_list[c][1]
                mf = combo_list[c][2]
                cw = combo_list[c][3]
                inner_iter += 1
                print('Iteration #', inner_iter)

                dtree_model = DecisionTreeClassifier(
                    criterion=crt, splitter=sp, max_features=mf, class_weight=cw
                )

                dtree_model.fit(train_X, train_Y)
                res_dtree = dtree_model.predict(test_X)
                score_dtree = accuracy_score(test_Y, res_dtree)

                print('With combo #{} , #{} Combinations Successful!'.format(
                    c+1, success+1))
                success += 1
                print(
                    'Dec_Tree_Classifier >> Comp:', compo+1, ' criterion:', crt,
                    ' splitter:', sp, ' max_features:', mf, ' class_weight:', cw, ' Score:', score_dtree, ' Highest Score:', sdt
                )
                if score_dtree > sdt:
                    print('New highest accuracy:', score_dtree, '>', sdt)
                    sdt = score_dtree

                    outSheet.write(line+1, 0, crt)
                    outSheet.write(line+1, 1, sp)
                    outSheet.write(line+1, 2, mf)
                    outSheet.write(line+1, 3, cw)
                    outSheet.write(line+1, 4, sdt)
                    outSheet.write(line+1, 5, compo+1)
                    print('At line {} writing file for component {}'.format(
                        line+1, compo+1))
                    line += 1
                    time.sleep(.5)
            except:
                print('Combo failed at', c+1)
                #fail += 1
        time.sleep(.5)

        os.system('cls')

    outWorkbook.close()
    print('All done!')
    '''print('Successful combos #{}'.format(len(successful_combos)))
    print('Counted successful combos %d'%success)
    print('Counted failed combos %d'%fail)
    print(successful_combos)'''


if __name__ == "__main__":
    decision_tree_classifier_func()
