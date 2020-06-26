import xlsxwriter as xl
import time

import numpy as np

from sklearn.model_selection import train_test_split

from functional_modules import file_locations_module as flocate
from functional_modules import pca_module

from functional_modules import DTreeModel as dtree
from functional_modules import GaussianNBmodel as gauss
from functional_modules import KNeighborModel as knbr
from functional_modules import LDAmodel as lda
from functional_modules import LogRegModel as log
from functional_modules import RandForestModel as rf
from functional_modules import SVCmodel as svc


def create_excel(excel_loc, title, classifier):
    headers = classifier.headers

    outWorkbook = xl.Workbook(excel_loc+title+'.xlsx')
    outSheet = outWorkbook.add_worksheet()
    L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(len(headers)):
        outSheet.write(L[i]+'1', headers[i])
    print('Excel created >> '+excel_loc+title+'.xlsx')
    return outWorkbook, outSheet


def prepare_data(data_path):
    all_data = np.load(data_path, allow_pickle=True)
    print('Data shape >> ', all_data.shape)
    all_X = all_data[:, :-1]
    all_Y = all_data[:, -1]
    print('Data distribution complete.')
    return all_X, all_Y


def train_model(classifier, X, Y, book, sheet, line=1, serial=1, doCompo=False):

    combo_list = classifier.combos  # call function here
    number_of_combos = len(combo_list)
    print('Number of combinations >> ', number_of_combos)
    print()

    headers = classifier.headers

    print('Processing compo #', serial)
    x = X
    if doCompo:
        x = pca_module.applyPCA(X, serial)
        print('PCA successfully applied for component #%d' % (serial+1))
    
    success = 0
    best_score = 0
    fail = 0
    scores = []

    train_X, test_X, train_Y, test_Y = train_test_split(x, Y, test_size=0.3)

    for c in range(number_of_combos):
        print('Entering combo #', c+1)
        try:
            score_model = classifier.make_model(c, train_X, train_Y, test_X, test_Y)
            #time.sleep(1)
            print('for Compo #{} - Combo #{} - #{} Combos Successful!\nScore: {}'.format(serial, c+1, success+1, score_model))
            success += 1
            if score_model > best_score:
                print('New highest accuracy:',score_model, '>', best_score)
                print(combo_list[c])
                best_score = score_model
                scores.append(best_score)
                limit = len(headers) - 3
                for i in range(len(headers)-3):
                    sheet.write(line, i, combo_list[c][i])

                sheet.write(line, (len(headers)-1), best_score*100)
                sheet.write(line, (len(headers)-2), serial)
                sheet.write(line, (len(headers)-3), best_score)
                print('Line #{} --- Component #{}'.format(line, serial))
                line += 1
        except:
            print('Compo #',serial,' - Combo failed at #', c+1)
            fail += 1
        print('Exiting compo #%d - combo #%d'% (serial, c+1))
        print()

    print('Compo %d - all done.'%serial)
    print('Total combinations: ', number_of_combos)
    print('Total success: ', success)
    print('Total failure:', fail)
    #input('ENTER to continue...')
    return line,scores


def classify_glcm(model, book, sheet, limit, path):
    X, Y = prepare_data(path)
    line = 1
    scores = []
    for serial in range(1,limit):
        line, best_scores = train_model(model, X, Y, book, sheet, line, serial, True)
        scores.append(best_scores)
        print('Serial #', serial, 'done.')
    print(scores)


def classify_hog(model, book, sheet, limit):    
    line = 1
    scores = []
    for serial in range(1,limit):
        path = flocate.HOG_all_case_feats_form.format(serial)
        X, Y = prepare_data(path)
        line, best_scores = train_model(model, X, Y, book, sheet, line, serial, False)
        scores.append(best_scores)
        print('Serial #', serial, 'done.')
    print(scores)


def classify_vlad(model, book, sheet, path):  
    X, Y = prepare_data(path)
    scores = []
    line, scores = train_model(model, X, Y, book, sheet)
    print(scores)
    return


if __name__ == "__main__":
    start_time = time.time()

    model = dtree
    #model = gauss
    #model = knbr
    #model = svc
    #model = rf     # time consuming - 36 combos
    #model = lda    # time consuming - 210 combos
    #model = log     # time consuming - 336 //924 combos
    
    #title = model.title+'_glcm'
    #title = model.title+'_hog'
    title = model.title +'_vlad50'

    excel_loc = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\excels\\'
    book, sheet = create_excel(excel_loc, title, model)

    limit = 161

    # function for handling glcm
    glcm_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\GLCM54feats54.npy"
    #classify_glcm(model, book, sheet, limit, glcm_path)
    
    # function for handling hog
    #classify_hog(model, book, sheet, limit)

    # function for handling vlad
    vlad50_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\vlad50_all_cases.npy"
    classify_vlad(model, book, sheet)
    
    book.close()
    print()

    e = int(time.time() - start_time)
    print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))



'''
steps
------
1. create excel
2. Train, target split
3. train model
'''
