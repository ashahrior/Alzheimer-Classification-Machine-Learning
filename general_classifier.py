import xlsxwriter as xl

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


def create_excel(excel_loc,title,classifier):
    headers = classifier.headers

    outWorkbook = xl.Workbook(excel_loc+title+'.xlsx')
    outSheet = outWorkbook.add_worksheet()
    L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(len(headers)):
        outSheet.write(L[i]+'1',headers[i])
    print('Excel created >> '+excel_loc+'\\'+title+'.xlsx')
    return outWorkbook, outSheet


def prepare_data(data_path):
    all_data = np.load(data_path,allow_pickle=True)
    print('Data shape >> ', all_data.shape)
    col = all_data.shape[1]
    all_X = all_data[:,:col-1]
    all_Y = all_data[:,col-1]
    print('Data distribution complete.')
    return all_X, all_Y


def train_model(classifier, X, Y, book, sheet, doCompo=False, start=0, finish=1):

    combo_list = classifier.combos # call function here
    number_of_combos = len(combo_list)
    print('Number of combinations >> ',number_of_combos)
    print()
    success = 0
    
    line = 0

    headers = classifier.headers

    for compo in range(start,finish):
        print('Entering compo #',compo+1)
        x = X
        if doCompo:
            x = pca_module.applyPCA(X, compo+1)
            print('PCA successfully applied for component #%d'%(compo+1))
        best_score = 0

        train_X, test_X, train_Y, test_Y = train_test_split(x, Y, test_size=0.3)
        for c in range(number_of_combos):
            print('Entering combo #',c+1)
            try:
                score_model = classifier.make_model(c, train_X, train_Y, test_X, test_Y)
                print('compo #{} - combo #{} - #{} Combinations Successful!'.format(compo+1, c+1, success+1))
                success += 1
                if score_model > best_score:
                    print('New highest accuracy:', score_model, '>', best_score)
                    print(combo_list[c])
                    best_score = score_model
                    for i in range(len(headers)-3):
                        sheet.write(line+1, i, combo_list[c][i])
                    
                    sheet.write(line+1,(len(headers)-1),best_score*100)
                    sheet.write(line+1,(len(headers)-2), compo+1)
                    sheet.write(line+1,(len(headers)-3), best_score)
                    print('Line #{} --- Component #{}'.format(line+1, compo+1))
                    line+=1
            except:
                print('Combo failed at #',c+1)
            print('Exiting combo #',c+1)
            print()
        print('Component #',compo+1,' complete.')
        print()
    book.close()            
    print('All done.')
    return


if __name__ == "__main__":
    
    serial = 1
    path = flocate.HOG_all_case_feats_form.format(serial)
    #path = flocate.GLCM_all_case_feats_file

    X,Y = prepare_data(path)
    
    excel_loc = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\ClassifierResults\\"
    model = dtree
    #model = gauss
    #model = knbr
    #model = lda
    #model = log
    #model = rf
    #model = svc
    
    #title = model.title+'glcm'
    title = model.title+'_hog'
    
    book, sheet = create_excel(excel_loc,title,model)
    
    train_model(model, X, Y, book, sheet, doCompo=False, start=0, finish=1)

'''
steps
------
1. provide path for source file location
2. prepare data by splitting
3. provide result storage location and title for excel
4. create excel
5. 
'''




    
