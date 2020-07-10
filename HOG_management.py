import os
import time

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import exposure
from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from functional_modules import data_viewer_module as dv

cases = ["AD", "CN", "MCI"]

resizeRow, resizeCol = 512, 256
resolution = f"{resizeRow}x{resizeCol}"
title = r"{}-HOGfeat-{}{}"

hog_feat_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\{}_iHOG\\"

hog_img_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\{}_HOG_img_"+resolution+"\\"

norm_npy_fold = "E:\\THESIS\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\INTEREST_NPY_DATA\\Normalized_NPY\{}_normNPY\\"

def calculate_HOG_feats():
    global resolution
    for cs in cases:
        os.chdir(norm_npy_fold.format(cs))
        for f in os.listdir():
            fname, fext = os.path.splitext(f)
            hog_file_name = title.format(fname, resolution, fext)
            print(f'\n{f} opened')
            data = np.load(f, allow_pickle=True)
            hog_feats = []
            #hog_images = []
            for slices in range(data.shape[0]):
                imageSlice = data[slices]

                resolution = str(resizeRow)+'x'+str(resizeCol)

                imageSliceResized = skimage.transform.resize(
                    imageSlice, (resizeRow, resizeCol))
                print(f'{fname} #{slices+1} slice >> resized -', end=' ')
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

            np.save(hog_feat_fold.format(cs) + hog_file_name, hog_feats)
            print(f'{hog_file_name} hog feat saved.')
    return


##### Applying PCA method
def apply_PCA(feature, no_comp):
    print('Applying PCA for #{} components'.format(no_comp))
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components=no_comp)
    pcomp = pca.fit_transform(X)

    return pcomp


def generate_HOG_array(case_type, number_of_files, target, no_comp, F):
    '''
    :param case_type: type of case being handled- AD/CN/MCI
    :param number_of_files: number of data files
    :param target: 1 for AD, 2 for CN and 3 for MCI
    :param no_comp: number of PCA
    :param F: Feature Array, just send a list
    :return: returns the updated feature list
    '''
    hog_feat_file_form = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_data\{}_HOG_256x128\256x128_hogFeat{}_data{}.npy"
    print('Inside generate_HOG_array() function with case type-{} for number of components-{}'.format(case_type, no_comp))
    print()

    for i in range(number_of_files):
        hog_data = np.load(hog_feat_file_form.format(
            case_type, case_type, i+1), allow_pickle=True)
        print('HOG data for', hog_feat_file_form.format(
            case_type, case_type, i+1), 'loaded.')

        comp = apply_PCA(hog_data, no_comp)
        print('PCA applied for {} case in file #{} for {} components.'.format(
            case_type, i+1, no_comp))

        row = []
        for j in range(111):
            for k in range(no_comp):
                row.append(comp[j][k])
        row.append(target)

        F.append(row)
        print('Row for case-%s for file #%d with %d components appended.' %
              (case_type, i+1, no_comp))
        print()
    return F


def merge_HOG_array(totalComp):
    '''
    :param totalComp: highest values of PCA
    :return:
    '''
    n_AD_file = 54  # Total datafiles of AD
    n_CN_file = 115
    n_MCI_file = 133

    hog_merged_file = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_merged\HOG_merged_feat{}.npy'

    start, end = 100, totalComp

    #start, end = 90, 100
    #start, end = 80, 90
    #start, end = 70, 80
    #start, end = 60, 70
    #start, end = 50, 60
    #start, end = 40, 50
    #start, end = 30, 40
    #start, end = 20, 30
    #start, end = 10, 20
    #start, end = 0, 10

    for i in range(start, end):

        F = []

        case_type = 'AD'
        print('Initiating AD for #{} components'.format(i+1))
        print()
        F = generate_HOG_array(case_type, n_AD_file, 1, i+1, F)
        print('HOG-AD feature list for #{} component stored.'.format(i+1))
        print()
        print()

        case_type = 'CN'
        print('Initiating CN for #{} components'.format(i+1))
        print()
        F = generate_HOG_array(case_type, n_CN_file, 2, i+1, F)
        print('HOG-CN feature list for #{} component stored.'.format(i+1))
        print()
        print()

        case_type = 'MCI'
        print('Initiating MCI for #{} components'.format(i+1))
        print()
        F = generate_HOG_array(case_type, n_MCI_file, 3, i+1, F)
        print('HOG-MCI feature list for #{} component stored.'.format(i+1))
        print()
        print()

        np.save(hog_merged_file.format(i+1), F)
        print('HOG merged feature list .npy for #{} component saved.'.format(i+1))
        print()
        print()
        os.system('cls')

    print('All The HOG Features Arrays saved Successfully.')


if __name__ == "__main__":
    pass
