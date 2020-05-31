import itertools
import os
import time
import xlsxwriter as xl

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import log_reg_classifier_combos as log_reg_combo
from functional_modules import pca_module

from sklearn.linear_model import LogisticRegression


def log_reg_classifier_func():
    # Feature of GLCM is Loaded.
    all_GLCM_features = np.load(
        flocate.GLCM_all_case_feats_file, allow_pickle=True)

    # The file contains 302 rows and 778 columns. The Last column is the target value.
    # all_X contains the 778 columns of features where each row represent a Data volume
    # all_Y contains the target value from the last column.
    all_X = all_GLCM_features[:, :777]
    all_Y = all_GLCM_features[:, 777]

    inner_iter = 0
    # number_principal_components = 300
    #successful_combos = []

    # A list of all possible parameters and their values collected from sklearn site
    # Paremeters with their possible values for LogReg

    penalty = ['l1', 'l2', 'elasticnet', 'none']
    dual = [False, True]
    tol = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002]
    fit_intercept = [False, True]
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    multiclass = ['auto', 'ovr', 'multinomial']
    warm = [True, False]

    parameters_list = [ penalty, dual, tol, fit_intercept, solver, multiclass, warm ]

    #combo_list = list(itertools.product(*parameters_list))
    combo_list = log_reg_combo.logistic_regression_combos

    number_of_combos = len(combo_list)
    # print(number_of_combos)
    # time.sleep(2)

    excel_loc = r"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\"

    #st, fin = 200, 300
    #st, fin = 100, 200
    st, fin = 0, 300

    outWorkbook = xl.Workbook(excel_loc+'log_reg_glcm_pca.xlsx')
    outSheet = outWorkbook.add_worksheet()

    outSheet.write('A1', 'PENALTY')
    outSheet.write('B1', 'DUAL')
    outSheet.write('C1', 'TOLERANCE')
    outSheet.write('D1', 'FIT-INTERCEPT')
    outSheet.write('E1', 'SOLVER')
    outSheet.write('F1', 'MULTICLASS')
    outSheet.write('G1', 'WARM-START')
    outSheet.write('H1', 'BEST ACCURACY')
    outSheet.write('I1', 'COMPONENT NO.')

    line = 0

    for i in range(st, fin):  # for the number principal components

        # get features with i+1 components

        all_p_comp = pca_module.applyPCA(all_X, i+1)

        train_X, test_X, train_Y, test_Y = train_test_split(
            all_p_comp, all_Y, test_size=0.3)  # Splitting Train, Test

        slg = 0

        for c in range(number_of_combos):
            try:
                # print('#{} combo attempted'.format(c))
                p = combo_list[c][0]
                d = combo_list[c][1]
                t = combo_list[c][2]
                fi = combo_list[c][3]
                s = combo_list[c][4]
                m = combo_list[c][5]
                w = combo_list[c][6]
                inner_iter += 1
                print('Iteration #', inner_iter)
                log_reg_model = LogisticRegression(
                    penalty=p, dual=d, tol=t,
                    fit_intercept=fi, max_iter=100,
                    random_state=0, solver=s,
                    multi_class=m, warm_start=w
                )
                log_reg_model.fit(train_X, train_Y)
                res_log_reg = log_reg_model.predict(test_X)
                score_log_reg = accuracy_score(test_Y, res_log_reg)
                print('#{} Combination Successful!'.format(c+1))
                # successful_combos.append([p,d,t,fi,s,m,w])
                print('Log_Reg >> Comp:', i+1, ' penalty:', p, ' Dual:', d,
                      ' Tol:', t, ' FI:', fi,
                      ' solver:', s, ' mulclass:', m,
                      ' warmStart:', w, ' Score:', score_log_reg, 'Highest Score', slg)

                if score_log_reg > slg:
                    print('New highest accuracy:', score_log_reg, '>', slg)
                    slg = score_log_reg
                    outSheet.write(line+1, 0, p)
                    outSheet.write(line+1, 1, d)
                    outSheet.write(line+1, 2, t)
                    outSheet.write(line+1, 3, fi)
                    outSheet.write(line+1, 4, s)
                    outSheet.write(line+1, 5, m)
                    outSheet.write(line+1, 6, w)
                    outSheet.write(line+1, 7, slg)
                    outSheet.write(line+1, 8, i+1)
                    print('Writing file for component {} at line {}'.format(i+1,line+1))
                    line+=1
                    time.sleep(.5)
                print()
            except:
                pass
        time.sleep(.3)
        os.system('cls')

    outWorkbook.close()


if __name__ == '__main__':
    log_reg_classifier_func()
