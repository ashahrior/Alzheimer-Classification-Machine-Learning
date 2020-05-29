import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import exposure
from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize

import data_viewer as dv
import feature_computation_module as fc
import file_locations_module as flocate 

'''
Select source location of file type to work with. AD / CN / MCI
'''
#src = flocate.npy_main_AD + 'data{}.npy'
#src = flocate.npy_main_CN +  'data{}.npy'
src = flocate.npy_main_MCI + 'data{}.npy'

# location for saving the hog features
# Remember to handle the print format
hog_feat_target_location = "E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_data\{}_HOG_256x128\{}_hogFeat{}_data{}"
# location for saving the hog images
# Remember to handle the print format
hog_img_target_location = "E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_data\{}_HOG_256x128\{}_hogImg{}_data{}"

# Select cast type for target location.
#case_type = 'AD' #done
#case_type = 'CN'
case_type = 'MCI'
resizeRow, resizeCol = 256, 128

# AD - 54 DONE
# CN - 115
# MCI - 133
# File iteration number. Change it after intervals.
from_file, to_file = 101, 134


for file_serial in range(from_file, to_file):
    break # remove the break to work
    file = fc.open_interest_data(src, file_serial)

    hog_images = []  # list of slices in a volume
    hog_feats = []  # list of feats per slice in a volume

    for slices in range(file.shape[0]):
        imageSlice = file[slices]
    
        resolution = str(resizeRow)+'x'+str(resizeCol)
        #resizeRow, resizeCol = 512, 256

        imageSliceResized = skimage.transform.resize(
            imageSlice, (resizeRow, resizeCol))
        print("{}- File #{} Slice #{}: Resized.".format(case_type, file_serial, slices))
        
        fd, hog_img = hog(
            imageSliceResized, orientations=9,
            pixels_per_cell=(8,8), cells_per_block=(2, 2),
            visualize=True, multichannel=False
        )
        print('{}- File #{} Slice #{}: HOG feat desc and image generated.'.format(case_type, file_serial, slices))

        hog_feats.append(fd)
        hog_images.append(hog_img)

    hog_feats = np.asarray(hog_feats)
    hog_images = np.asarray(hog_images)
    dv.Show(hog_images)
    break

    np.save(hog_feat_target_location.format( case_type, resolution, case_type, file_serial), hog_feats )
    print(hog_feat_target_location.format( case_type, resolution, case_type, file_serial), "- Saved" )
    
    '''
    np.save(hog_img_target_location.format(
        case_type, resolution, case_type, file_serial), hog_images)
    print(hog_img_target_location.format(
        case_type, resolution, case_type, file_serial), "- Saved")
    '''
    print()

'''
    plt.figure('Histogram of Oriented Gradients')

    slice = 70
    plt.subplot(121), plt.imshow(file[slice], cmap=plt.cm.gray), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(hog_images[slice], cmap=plt.cm.gray), plt.title('HOG Feats')
    plt.xticks([]), plt.yticks([])

    #plt.savefig('E:\THESIS\CODING\FinalCodingDefence\Images\img{}x{}.png'.format(resizeRow,resizeCol),dpi=1600)
    plt.show()
    hog_images = np.load("E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\HOG_data\\AD_HOG\\256x128_hogImgAD_T43.npy",allow_pickle = True)

    dv.Show(hog_images)
 '''