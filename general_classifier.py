import xlsxwriter as xl

import numpy as np

from sklearn.model_selection import train_test_split

from functional_modules import HOG_all_cases_feats_form
from functional_modules import pca_module
from functional_modules import DTreeModel as dtree
from functional_modules import GaussianNBmodel as gauss



def create_excel(excel_loc,title,headers):
    outWorkbook = xl.Workbook(excel_loc+title)
    outSheet = outWorkbook.add_worksheet()
    L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(len(headers)):
        outSheet.write(L[i]+'1',headers[i])
    return outWorkbook, outSheet


def prepare_data(data_path):
    all_data = np.load(data_path,allow_pickle=True)

    col = all_data.shape[1]

    all_X = all_data[:,:col-1]
    all_Y = all_data[:,col-1]

    return all_X, all_Y


def train_model(classifier, X, Y, book, sheet, doCompo=False, start=0, finish=1):

    combo_list = classifier.combos # call function here
    number_of_combos = len(combo_list)

    success = 0
    line = 0

    headers = classifier.headers

    for compo in range(start,finish):
        if doCompo:
            X = pca_module.applyPCA(X, compo+1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

        best_score = 0
        for c in range(number_of_combos):
            try:
                score_model = classifier.make_model(c, train_X, train_Y, test_X, test_Y)
                print('With combo #{} , #{} Combinations Successful!'.format(c+1, success+1))
                success += 1
                if score_model > best_score:
                    print('New highest accuracy:', score_model, '>', best_score)
                    best_score = score_model
                    for i in range(len(headers)):
                        sheet.write(line+1, i, combo_list[c][i])
                        print('Line #{} --- Component #{}'.format(line+1, compo+1))
            except:
                print('Combo failed at #',c+1)
            book.close()
            book.close()
            print('All done.')
    return


if __name__ == "__main__":
    
    serial = 1
    path = HOG_all_cases_feats_form.format(serial)

    X,Y = prepare_data(path)







    
