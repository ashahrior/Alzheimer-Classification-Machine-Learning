import math
import os

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

fc.load_data(flocate.nii_AD_path, flocate.npy_main_AD_path,0)
fc.load_data(flocate.nii_CN_path, flocate.npy_main_CN_path,0)
fc.load_data(flocate.nii_MCI_path, flocate.npy_main_MCI_path,0)


# Calculating the GLCM Features

fglcm.calculate_GLCM_feats(flocate.npy_main_AD_path, flocate.glcm_AD_path,54)
fglcm.calculate_GLCM_feats(flocate.npy_main_CN_path, flocate.glcm_CN_path, 115)
fglcm.calculate_GLCM_feats(flocate.npy_main_MCI_path, flocate.glcm_MCI_path, n=133)


# Creating a feature map for all the glcm address

F = [] #The Total List of the Features

F = fglcm.generate_GLCM_feats_list(flocate.glcm_AD_path,54,1,F)
print('AD-GLCM feature stored.')

F = fglcm.generate_GLCM_feats_list(flocate.glcm_CN_path,115,2,F)
print('CN-GLCM feature stored.')

F = fglcm.generate_GLCM_feats_list(flocate.glcm_MCI_path,133,3,F)
print('MCI-GLCM feature stored.')


# Saving the F as .npy array
fc.np.save(flocate.npy_GLCM_feature_path+'GLCM_all_cases_{}.npy'.format(1),F)
