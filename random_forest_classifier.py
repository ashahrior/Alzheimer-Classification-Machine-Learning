import itertools
import os
import time
import xlsxwriter as xl

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import pca_module

from sklearn.ensemble import RandomForestClassifier

####### Main Program .... #####


def rand_forest_classifier_func():
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
    criterion = ['gini', 'entropy']
    max_features = ['auto', 'sqrt', 'log2']
    bootstrap = [True, False]
    oob_score = [True, False]
    warm_start = [True, False]
    '''
    parameters_list = [ criterion, max_features, bootstrap, oob_score, warm_start]
    feature_combo = list(itertools.product(*parameters_list))
    # there are total 48 combinations originally
    feature_combo = [list(x) for x in feature_combo]
    '''
    feature_combo = [
        ['gini', 'auto', True, True, True], ['gini', 'auto', True, True, False], ['gini', 'auto', True, False, True], ['gini', 'auto', True, False, False], ['gini', 'auto', False, False, True], ['gini', 'auto', False, False, False], ['gini', 'sqrt', True, True, True], ['gini', 'sqrt', True, True, False], ['gini', 'sqrt', True, False, True], ['gini', 'sqrt', True, False, False], ['gini', 'sqrt', False, False, True], ['gini', 'sqrt', False, False, False], ['gini', 'log2', True, True, True], ['gini', 'log2', True, True, False], ['gini', 'log2', True, False, True], ['gini', 'log2', True, False, False], ['gini', 'log2', False, False, True], ['gini', 'log2', False, False, False], ['entropy', 'auto', True, True, True], ['entropy', 'auto', True, True, False], ['entropy', 'auto', True, False, True], ['entropy', 'auto', True, False, False], ['entropy', 'auto', False, False, True], ['entropy', 'auto', False, False, False], ['entropy', 'sqrt', True, True, True], ['entropy', 'sqrt', True, True, False], ['entropy', 'sqrt', True, False, True], ['entropy', 'sqrt', True, False, False], ['entropy', 'sqrt', False, False, True], ['entropy', 'sqrt', False, False, False], ['entropy', 'log2', True, True, True], ['entropy', 'log2', True, True, False], ['entropy', 'log2', True, False, True], ['entropy', 'log2', True, False, False], ['entropy', 'log2', False, False, True], ['entropy', 'log2', False, False, False]
        ]


    number_of_combos = len(feature_combo)  # 36
    '''
    st, fin = 0, 100  # 300
    serial = 1
    '''
    st, fin = 100, 200
    serial = 3
    '''
    st, fin = 200, 300
    serial = 2'''
    success = 0
    fail = 0

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"

    inner_iter = 0
    line = 0

    #success_list = []

    outWorkbook = xl.Workbook(excel_loc+'rand_forest_glcm_pca{}.xlsx'.format(serial))
    outSheet = outWorkbook.add_worksheet()

    outSheet.write('A1', 'CRITERION')
    outSheet.write('B1', 'MAX-FEATS')
    outSheet.write('C1', 'BOOTSTRAP')
    outSheet.write('D1', 'OOB-SCORE')
    outSheet.write('E1', 'WARM-START')
    outSheet.write('F1', 'BEST-ACCURACY')
    outSheet.write('G1', 'COMPONENT-NO.')
    outSheet.write('M1', 'ROW.')  # 12
    outSheet.write('N1', 'HIGHEST-ACCURACY')
    outSheet.write('O1', 'COMPONENT-NO.')

    highest_accuracy = 0
    final_row = 0
    cc = 0

    for compo in range(st, fin):
        all_p_comp = pca_module.applyPCA(all_X, compo+1)

        train_X, test_X, train_Y, test_Y = train_test_split(all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test
        srf = 0  # random forest accuracy

        for c in range(number_of_combos):
            try:
                C = feature_combo[c][0]
                MF = feature_combo[c][1]
                B = feature_combo[c][2]
                OOB = feature_combo[c][3]
                W = feature_combo[c][4]

                inner_iter += 1
                print('Iteration #', inner_iter)

                ran_forest_model = RandomForestClassifier(criterion=C, max_features=MF, bootstrap=B, oob_score=OOB, warm_start=W)
                ran_forest_model.fit(train_X, train_Y)
                res_ran_forest = ran_forest_model.predict(test_X)
                score_ran_forest = accuracy_score(test_Y, res_ran_forest)

                print('With combo #{} , #{} Combinations Successful!'.format(
                    c+1, success+1))
                success += 1
                #success_list.append([C, MF, B, OOB, W])
                print(
                    '~ RandomForest_Classifier >> Comp:', compo +
                    1, '\ncriterion: ', C, ' --max_features: ', MF, '\nbootstrap:', B, ' --oob_score', OOB, ' --warm_start:', W, '\nScore:', score_ran_forest, ' ++Highest Score:', srf
                )
                print()
                if score_ran_forest > srf:
                    print('New highest accuracy:', score_ran_forest, '>', srf)
                    srf = score_ran_forest
                    outSheet.write(line+1, 0, C)
                    outSheet.write(line+1, 1, MF)
                    outSheet.write(line+1, 2, B)
                    outSheet.write(line+1, 3, OOB)
                    outSheet.write(line+1, 4, W)
                    outSheet.write(line+1, 5, srf)
                    outSheet.write(line+1, 6, compo+1)
                    print(
                        '+++ +++ At line {} writing file for component {}'.format(line+1, compo+1))
                    line += 1
                    print()
                    print()
            except:
                print('* Combo failed at', c+1)
                print()
                fail += 1

        if srf > highest_accuracy:
            highest_accuracy = srf
            cc = compo+1
            final_row = line+1
            print('### ### New highest', highest_accuracy, ' @', cc)
            print()
            print()

        # os.system('cls')
    print()
    outSheet.write(1, 12, final_row)
    outSheet.write(1, 13, highest_accuracy)
    outSheet.write(1, 14, cc)
    print('Writing highest accuracy {} for component #{}'.format(highest_accuracy, cc))
    print()

    outWorkbook.close()
    print('All done!')
    #print('Counted successful combos %d' % success)
    #print('Counted failed combos %d' % fail)


if __name__ == "__main__":
    rand_forest_classifier_func()
