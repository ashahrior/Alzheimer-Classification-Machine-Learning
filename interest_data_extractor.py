import numpy as np
import os
import re
from functional_modules import data_viewer_module as dv

ad_data_tuples = {
    1: (54, 112), 2: (68, 121), 3: (74, 125), 4: (74, 124), 5: (64, 118), 6: (82, 132), 7: (74, 124), 8: (74, 128), 9: (72, 122), 10: (72, 125), 11: (56, 110), 12: (59, 113), 13: (68, 124), 14: (70, 124), 15: (65, 118), 16: (68, 118), 17: (91, 141), 18: (62, 108), 19: (68, 122), 20: (72, 126), 21: (46, 108), 22: (72, 128), 23: (82, 124), 24: (54, 110), 25: (84, 142), 26: (70, 118), 27: (60, 114), 28: (70, 114), 29: (92, 145), 30: (64, 116), 31: (56, 106), 32: (68, 120), 33: (71, 117), 34: (78, 124), 35: (64, 117), 36: (84, 140), 37: (74, 134), 38: (54, 106), 39: (54, 106), 40: (74, 134), 41: (82, 138), 42: (61, 128), 43: (68, 128), 44: (46, 110), 45: (82, 127), 46: (62, 130), 47: (100, 146), 48: (54, 114), 49: (52, 96), 50: (80, 126), 51: (84, 134), 52: (68, 114), 53: (66, 114), 54: (80, 140)
}

src = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\{}_mainNPY"

targ = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\{}_iNPY"

def get_img_names():
    titles = ''
    for file in os.listdir():
        if file.__contains__('.jpg'):
            titles += file+'\n'
    return titles


def get_tuples(p, case, n):
    tuples = {}

    for i in range(1, n+1):
        path = p.format(case, case, i)
        os.chdir(path)
        imgs = get_img_names()
        serial = re.findall(r'g[1-9]*[0-9][0-9]', imgs)
    #print(serial)
        serial = [int(i[1:]) for i in serial]
        a, b = min(serial), max(serial)
        #print(i, f'- ({a},{b})')
        tuples[i] = (a,b)
    return tuples


def get_interest_data(src, targ, case, n):
    file = src.format(case) + '\data{}.npy'
    for i in range(1, n + 1):
        data = np.load(file.format(i), allow_pickle=True)
        tup = ad_data_tuples[i]
        i_data = data[tup[0] : tup[1] + 1, :, :]
        print(f'{case} Data-{i} with interest space {tup} loaded')
        #dv.Show(i_data)
        output = targ.format(case) + f'\data{i}.npy'
        np.save(output, i_data)
        print(f'{case} Data-{i} with interest space {tup} saved')
    return


if __name__ == '__main__':

    case = 'AD'

    n = 54

    p = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\IMAGES\{}\{}-Data{}"

    #get_interest_data(src, targ, case,n)

    #AD = get_tuples(p, case, n)
    '''print(AD)
    for key,values in AD.items():
        print(key,'-',values)'''
