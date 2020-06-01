import itertools
import os
import time
import xlsxwriter as xl

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import pca_module

from sklearn.neighbors import KNeighborsClassifier

####### Main Program .... #####


def kNeighbor_classifier_func():
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
    weights = ['uniform', 'distance']
    algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']

    parameters_list = [weights, algorithm]
    combo_list = list(itertools.product(*parameters_list))

    combo_list = [list(x) for x in combo_list]

    number_of_combos = len(combo_list)

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"

    inner_iter = 0
    line = 0

    st, fin = 0, 300
    success = 0
    fail = 0
    

    outWorkbook = xl.Workbook(excel_loc+'kNeighbors_glcm_pca.xlsx')
    outSheet = outWorkbook.add_worksheet()

    outSheet.write('A1', 'WEIGHTS')
    outSheet.write('B1', 'ALGORITHM')
    outSheet.write('C1', 'BEST-ACCURACY')
    outSheet.write('D1', 'COMPONENT-NO.')
    outSheet.write('J1', 'ROW.')
    outSheet.write('K1', 'HIGHEST-ACCURACY')
    outSheet.write('L1', 'COMPONENT-NO.')
    
    highest_accuracy = 0
    final_row = 0
    cc = 0

    for compo in range(st, fin):
        all_p_comp = pca_module.applyPCA(all_X, compo+1)

        train_X, test_X, train_Y, test_Y = train_test_split(
            all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test
        skn = 0  # KNeighbours

        for c in range(number_of_combos):
            try:
                W = combo_list[c][0]
                A = combo_list[c][1]
                inner_iter += 1
                print('Iteration #', inner_iter)

                kneighbors_model = KNeighborsClassifier(weights=W,algorithm=A)
                kneighbors_model.fit(train_X,train_Y)
                res_kneighbors = kneighbors_model.predict(test_X)
                score_kneighbors = accuracy_score(test_Y,res_kneighbors)

                print('With combo #{} , #{} Combinations Successful!'.format(c+1, success+1))
                success += 1
                print(
                    'KNeighbor_Classifier >> Comp:', compo+1, ' Weights: ', W, ' Algorithm: ', A, ' Score:', score_kneighbors , ' Highest Score:', skn
                )
                if score_kneighbors > skn:
                    print('New highest accuracy:', score_kneighbors, '>', skn)
                    skn = score_kneighbors
                    outSheet.write(line+1, 0, W)
                    outSheet.write(line+1, 1, A)
                    outSheet.write(line+1, 2, skn)
                    outSheet.write(line+1, 3, compo+1)
                    print('At line {} writing file for component {}'.format(line+1, compo+1))
                    line += 1
                    print()
            except:
                print('Combo failed at', c+1)
                fail += 1

        if skn > highest_accuracy:
            highest_accuracy = skn
            cc = compo+1
            final_row = line+1
            print('New highest',highest_accuracy,' @',cc)
            print()

        #os.system('cls')
    print()
    outSheet.write(1, 9, final_row)
    outSheet.write(1, 10, highest_accuracy)
    outSheet.write(1,11, cc)
    print('Writing highest accuracy {} for component #{}'.format(highest_accuracy,cc))
    print()

    outWorkbook.close()
    print('All done!')
    print('Counted successful combos %d'%success)
    print('Counted failed combos %d'%fail)


if __name__ == "__main__":
    kNeighbor_classifier_func()
