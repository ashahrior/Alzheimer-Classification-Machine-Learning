import os
import time

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import exposure
from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize

from functional_modules import data_viewer_module as dv

case_type = 'MCI'  # 'CN' # 'AD'

resizeRow, resizeCol = 512, 256
resolution = f"{resizeRow}x{resizeCol}"

tit = r"{}-HOGfeat-{}{}"

hog_feat_file_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\{}_iHOG\\"

hog_img_targ_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\{}_HOG_img_"+resolution+"\\"

srcfol = f"E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\INTEREST_NPY_DATA\\Normalized_NPY\{case_type}_normNPY\\"

start_time = time.time()

os.chdir(srcfol)

for f in os.listdir():
    fname, fext = os.path.splitext(f)
    targ_file = tit.format(fname, resolution, fext)
    print(f'\n{f} opened')
    data = np.load(f, allow_pickle=True)

    hog_feats = []
    #hog_images = []

    for slices in range(data.shape[0]):
        imageSlice = data[slices]

        resolution = str(resizeRow)+'x'+str(resizeCol)

        imageSliceResized = skimage.transform.resize(
            imageSlice, (resizeRow, resizeCol))
        print(f'#{slices+1} slice >> resized -', end=' ')
        m, n = 8, 2
        ppc = (m, m)
        cpb = (n, n)
        fd, hog_img = hog(
            imageSliceResized, orientations=9,
            pixels_per_cell=ppc, cells_per_block=cpb,
            visualize=True, multichannel=False
        )
        print('HOG feat generated')
        hog_feats.append(fd)
        #hog_images.append(hog_img)
    hog_feats = np.asarray(hog_feats)

    #hog_images = np.asarray(hog_images)
    #dv.Show(hog_images)
    #break
    
    np.save(hog_feat_file_path.format(case_type) + targ_file, hog_feats)
    print(f'{targ_file} hog feat saved.')

e = int(time.time() - start_time)
print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))


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
