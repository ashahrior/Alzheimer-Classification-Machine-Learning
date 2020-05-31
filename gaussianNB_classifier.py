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

from sklearn.naive_bayes import GaussianNB

####### Main Program .... #####


def gaussianNB_classifier_func():
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

    var_smoothing = 0.000000001

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"

    inner_iter = 0
    line = 0

    st, fin = 0, 300

    outWorkbook = xl.Workbook(excel_loc+'gaussNB_glcm_pca.xlsx')
    outSheet = outWorkbook.add_worksheet()

    outSheet.write('A1', 'VAR-SMOOTHING')
    outSheet.write('B1', 'BEST-ACCURACY')
    outSheet.write('C1', 'COMPONENT-NO.')
    outSheet.write('J1', 'ROW')
    outSheet.write('K1', 'HIGHEST-ACCURACY')
    outSheet.write('L1', 'COMPONENT-NO.')

    x = 0   # to keep line-track of highest score for any component
    success = 0
    fail = 0

    highest_accuracy = 0
    final_row = 0
    cc = 0

    for compo in range(st, fin):
        all_p_comp = pca_module.applyPCA(all_X, compo+1)

        train_X, test_X, train_Y, test_Y = train_test_split(all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test
        sgnb = 0  # GaussianNB

        try:
            var = var_smoothing
            inner_iter += 1
            print('Iteration #', inner_iter)

            gaussnb_model = GaussianNB(var_smoothing=var)
            gaussnb_model.fit(train_X, train_Y)
            res_gauss = gaussnb_model.predict(test_X)
            score_gauss = accuracy_score(test_Y, res_gauss)
            
            print(
                'GaussianNB_Classifier >> Comp:', compo+1, 
                ' var_smoothing:', var, ' Score:', score_gauss
            )

            print('Accuracy:', score_gauss, ' @', compo+1)
            sgnb = score_gauss
            outSheet.write(line+1, 0, var)
            outSheet.write(line+1, 1, sgnb)
            outSheet.write(line+1, 2, compo+1)
            print('At line {} writing file for component {}'.format(line+1, compo+1))
            line += 1
            print()
        except:
            print('Combo failed at', compo+1)
            fail += 1
        
        if sgnb > highest_accuracy:
            highest_accuracy = sgnb
            cc = compo+1
            final_row = line+1
            print('New highest',highest_accuracy,' @',cc)

    outSheet.write(1, 9, final_row)
    outSheet.write(1, 10, highest_accuracy)
    outSheet.write(1,11, cc)
    print('Writing highest accuracy for {}'.format(compo+1))

    outWorkbook.close()
    print('All done!')
    print()


if __name__ == "__main__":
    gaussianNB_classifier_func()
