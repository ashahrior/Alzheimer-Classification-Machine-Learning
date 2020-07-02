import math
import os
import time

import cv2
import nibabel as nib
import numpy as np

from functional_modules import data_viewer_module as dv
from functional_modules import feature_computation_module as fc
from functional_modules import feature_GLCM_module as fglcm
from functional_modules import file_locations_module as flocate


from functional_modules import data_viewer_module as dv
from functional_modules import feature_computation_module as fc
from functional_modules import feature_GLCM_module as fglcm

import numpy as np
import pandas as pd
import time

#case = 'AD'
#case = 'CN'
case = 'MCI'


def check():
    for n in range(1, 11):
        print(n)
        f = f"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\{case}_main_10_npy\data{n}.npy"
        d = np.load(f, allow_pickle=True)
        dv.Show(d)


def fetch():

    adx = [
        (56, 110), (56, 110), (63, 117), (87, 141), (90, 144), (82,
                                                                136), (100, 144), (56, 110), (60, 90), (84, 124)
    ]

    cnx = [
        (60, 114), (60, 114), (66, 100), (70, 114), (64, 90), (64,
                                                               100), (64, 100), (64, 100), (64, 110), (84, 128)
    ]

    mcix = [
        (64, 118), (64, 118), (74, 128), (78, 132), (50,
                                                     104), (64, 118), (64, 118), (54, 88), (64, 118), (54, 90)
    ]

    s = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\{}_mainNPY\\data{}.npy"
    t = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\{}_i10_npy\\"
    n = 1
    for val in mcix:
        d = np.load(s.format(case, n), allow_pickle=True)
        data = d[val[0]: val[1]+1]
        print(n, ' {} data - range'.format(case), val)
        print(data.shape)
        np.save(t.format(case) + 'data{}.npy'.format(n), data)
        n += 1
        print()


def glcm_calc():
    ads = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\ad_i10_npy\\'
    adt = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\glcm_10\ad_10\\'
    cns = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\cn_i10_npy\\'
    cnt = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\glcm_10\cn_10\\'
    mcis = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\mci_i10_npy\\'
    mcit = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\glcm_10\mci_10\\'
    fglcm.calculate_GLCM_feats(ads, adt, 10)
    fglcm.calculate_GLCM_feats(cns, cnt, 10)
    fglcm.calculate_GLCM_feats(mcis, mcit, 10)
    return


def glcm_merge():
    F = []  # The Total List of the Features
    print('AD initiated')
    #F = fglcm.generate_GLCM_feats_list(flocate.glcm_AD_path,54,1,F)
    ad_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\glcm_10\\ad_10\\'
    F = fglcm.generate_GLCM_feats_list(ad_path, 10, 1, F)
    print('AD-GLCM feature stored.\n')

    input('ENTER to initiate CN')
    cn_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\glcm_10\cn_10\\'
    F = fglcm.generate_GLCM_feats_list(cn_path, 10, 2, F)
    print('CN-GLCM feature stored.\n')

    input('ENTER to initiate MCI')
    mci_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\glcm_10\\mci_10\\'
    F = fglcm.generate_GLCM_feats_list(mci_path, 10, 3, F)
    print('MCI-GLCM feature stored.\n')

    # Saving the F as .npy array
    input('ENTER to save')
    glcm10_feats = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\TEN10\\'
    fc.np.save(glcm10_feats+'glcm_{}__feats.npy'.format(10), F)


def merge(F, case_type):
    if case_type == 'ad':
        target = 1
    elif case_type == 'cn':
        target = 2
    elif case_type == 'mci':
        target = 3
    merger = []
    folder_path = r"E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\TEN10\\glcm_10\\{}_10\\"
    merger = []
    for serial in range(1, 11):
        g = ['asm', 'brtns', 'diss', 'entropy', 'homo', 'idm', 'variance']
        row = []
        for i in g:
            file_path = folder_path.format(
                case_type) + i + '{}.npy'.format(serial)
            data = np.load(file_path, allow_pickle=True)
            print(case_type, '->', i + '{}.npy'.format(serial),
                  ' ---> ', data.shape)
            print(data)
            for j in range(data.shape[0]):
                row.append(j)
        row.append(target)
        F.append(row)
        return
    return F


def handle_missing_values():
    folder_path = r"E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\TEN10\\glcm_10\\{}_10\\"
    folders = ['ad', 'cn', 'mci']
    for f in folders:
        for serial in range(1, 11):
            g = ['asm', 'brtns', 'diss', 'entropy', 'homo', 'idm', 'variance']
            row = []
            for i in g:
                file_path = folder_path.format(f) + i + '{}.npy'.format(serial)
                data = np.load(file_path, allow_pickle=True)
                if 0 in data:
                    print('Found in', end=' ')
                    print(f, ' -> ', i + '{}.npy'.format(serial))
                    data = np.where(data == 0, np.nan, data)
                    dataframe = pd.Series(data)
                    interpolated_dataframe = dataframe.interpolate(
                        method='pchip')
                    print('Interpolation complete')
                    interpolated_numpy = interpolated_dataframe.to_numpy()
                    np.save(file_path, interpolated_numpy)
                    print(file_path+' saved.')
                    print()


start_time = time.time()
# check()
# fetch()
# glcm_calc()
# handle_missing_values()
#
# glcm_merge()
e = int(time.time() - start_time)
print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))


'''
Call fc.Load_Data 3 times for AD, CN, and MCI respectively
Call runAllFeature 3 times for AD, CN, and MCI respectively
Call featureArray 3 times for AD, CN, and MCI respectively and 
provide the same F to get The Features together
'''


##### Main Program #####


# Saving the .npy file from the nifti file format
'''
fc.convert_nii_to_npy(flocate.nii_AD_path, flocate.npy_main_AD_path,0)
fc.convert_nii_to_npy(flocate.nii_CN_path, flocate.npy_main_CN_path,0)
fc.convert_nii_to_npy(flocate.nii_MCI_path, flocate.npy_main_MCI_path,0)
'''

# Calculating the GLCM Features
'''
fglcm.calculate_GLCM_feats(flocate.npy_main_AD_path, flocate.glcm_AD_path,54)
fglcm.calculate_GLCM_feats(flocate.npy_main_CN_path, flocate.glcm_CN_path, 115)
fglcm.calculate_GLCM_feats(flocate.npy_main_MCI_path, flocate.glcm_MCI_path, n=133)
'''

# Creating a feature map for all the glcm address

start_time = time.time()

F = []  # The Total List of the Features
print('AD initiated')
#F = fglcm.generate_GLCM_feats_list(flocate.glcm_AD_path,54,1,F)
ad_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\glcm54_ad\\'
F = fglcm.generate_GLCM_feats_list(ad_path, 54, 1, F)
print('AD-GLCM feature stored.\n')

input('ENTER to initiate CN')
cn_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\glcm54_cn\\'
F = fglcm.generate_GLCM_feats_list(cn_path, 54, 2, F)
print('CN-GLCM feature stored.\n')

input('ENTER to initiate MCI')
mci_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\glcm54_mci\\'
F = fglcm.generate_GLCM_feats_list(mci_path, 54, 3, F)
print('MCI-GLCM feature stored.\n')


# Saving the F as .npy array
input('ENTER to save')
glcm54_feats = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\\'
fc.np.save(glcm54_feats+'GLCM54feats{}.npy'.format(54), F)

e = int(time.time() - start_time)
print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
