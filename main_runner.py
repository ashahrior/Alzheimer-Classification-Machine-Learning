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


'''
Call fc.Load_Data 3 times for AD, CN, and MCI respectively
Call runAllFeature 3 times for AD, CN, and MCI respectively
Call featureArray 3 times for AD, CN, and MCI respectively and 
provide the same F to get The Features together
'''


##### Main Program #####


#Saving the .npy file from the nifti file format
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

F = [] #The Total List of the Features
print('AD initiated')
#F = fglcm.generate_GLCM_feats_list(flocate.glcm_AD_path,54,1,F)
ad_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\glcm54_ad\\'
F = fglcm.generate_GLCM_feats_list(ad_path,54,1,F)
print('AD-GLCM feature stored.\n')

input('ENTER to initiate CN')
cn_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\glcm54_cn\\'
F = fglcm.generate_GLCM_feats_list(cn_path,54,2,F)
print('CN-GLCM feature stored.\n')

input('ENTER to initiate MCI')
mci_path = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\glcm54_mci\\'
F = fglcm.generate_GLCM_feats_list(mci_path,54,3,F)
print('MCI-GLCM feature stored.\n')


# Saving the F as .npy array
input('ENTER to save')
glcm54_feats = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\FiftyFour\\'
fc.np.save(glcm54_feats+'GLCM54feats{}.npy'.format(54),F)

e = int(time.time() - start_time)
print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
