import math
import os

import cv2
import nibabel as nib
import numpy as np

import data_viewer as dv
import feature_computation_module as fc
import feature_GLCM_module as fglcm
import file_locations_module as flocate


'''
Call fc.Load_Data 3 times for AD, CN, and MCI respectively
Call runAllFeature 3 times for AD, CN, and MCI respectively
Call featureArray 3 times for AD, CN, and MCI respectively and 
provide the same F to get The Features together
'''


##### Main Program #####


#Saving the .npy file from the nifti file format

fc.load_data(flocate.nii_AD, flocate.npy_main_AD,0)
fc.load_data(flocate.nii_CN, flocate.npy_main_CN,0)
fc.load_data(flocate.nii_MCI, flocate.npy_main_MCI,0)


# Calculating the GLCM Features

fglcm.calculate_GLCM_feats(flocate.npy_main_AD, flocate.glcm_AD,54)
fglcm.calculate_GLCM_feats(flocate.npy_main_CN, flocate.glcm_CN, 115)
fglcm.calculate_GLCM_feats(flocate.npy_main_MCI, flocate.glcm_MCI, n=133)


# Creating a feature map for all the glcm address

F = [] #The Total List of the Features

F = fglcm.generate_GLCM_feats_list(flocate.glcm_AD,54,1,F)
print('AD-GLCM feature stoed.')

F = fglcm.generate_GLCM_feats_list(flocate.glcm_CN,115,2,F)
print('CN-GLCM feature stored.')

F = fglcm.generate_GLCM_feats_list(flocate.glcm_MCI,133,3,F)
print('MCI-GLCM feature stored.')


# Saving the F as .npy array
#fc.np.save(flocate.npy_GLCM_feature+'GLCM_all_cases_{}.npy'.format(1),F)
