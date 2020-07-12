import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

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

imp_hog_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\imputed_HOG\\"

limit = 69


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
    print('applying PCA #{}'.format(no_comp))
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components=no_comp)
    pcomp = pca.fit_transform(X)

    return pcomp


def get_max_slice(src):
    """get the maxmimum number of slices of any file for overall dataset

    Args:
        src ([str]): [source location of the data files]
    """
    cases = ['AD', 'CN', 'MCI']
    limit = 0
    for c in cases:
        os.chdir(src.format(c))
        print(f'\nInside {c}')
        for file in os.listdir():
            data = np.load(file)
            slices = data.shape[0]
            print(f'{file} - {slices}',end=' --- ')
            limit = max(slices, limit)
    return limit


def add_nan(data):
    """[append nan value to files]

    Args:
        data ([ndarray]): [source data file]

    Returns:
        [ndarray]: [numpy array with nan appended]
    """
    global limit
    print(data.shape[0])
    nan_data = np.zeros((limit - data.shape[0], data.shape[1]))
    nan_data[:] = np.NaN
    padded_data = np.vstack([data, nan_data])
    return padded_data


def interpolate_data(data):
    """[Interpolation of passed to replace null values]

    Args:
        data ([ndarray]): [a numpy array]

    Returns:
        [dataframe]: [an interpolated dataframe]
    """
    df = pd.DataFrame(data)
    return df.interpolate(method='spline', order=1)


def perform_data_generation(cases=['AD', 'CN', 'MCI']):
    global limit
    for c in cases:
        os.chdir(hog_feat_fold.format(c))
        count = 0
        for file in os.listdir():
            print(file, '->', end='')
            fname, fext = os.path.splitext(file)
            data = np.load(file, allow_pickle=True)

            nan_data = add_nan(data)
            print(f'{fname} NaN addition done.')

            interpolated_df = interpolate_data(nan_data)
            print(f'{fname} interpolation done.')

            interpolated_data = interpolated_df.to_numpy()
            case_folder = imp_hog_fold+f"{c}_impHOG\\"
            np.save(case_folder + f"{fname}-imp{fext}", interpolated_data)

            print(f"{fname}-imp{fext} saved.")
            count += 1

        print(f'{c} - {count} done')



def generate_HOG_array(case_type, no_comp):

    target = {
        'AD': 1, 'CN': 2, 'MCI': 3
    }
    
    case_folder = imp_hog_fold + f"{case_type}_impHOG\\"
    os.chdir(case_folder)
    
    print("Inside generate_HOG_array() - CASE-{} N_Comp-{}\n".format(case_type, no_comp))
    F = []
    for file in os.listdir():
        print(file)
        fname, fext = os.path.splitext(file)
        data = np.load(file, allow_pickle=True)
        comp = apply_PCA(data, no_comp)
        print(f'#{no_comp} comp applied',end='-')
        flat = comp.flatten()
        flat = np.append(flat, [target[case_type]])
        F.append(flat)
        print('appended')
    return np.vstack(F)


def merge_HOG_array():
    
    hog_merged_file = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\HOG_merged\HOG54_merged_feat{}.npy'
    
    global limit
    totalcomp = limit  #69
    
    start, end = 60, totalcomp
    #start, end = 50, 60
    #start, end = 40, 50
    #start, end = 30, 40
    #start, end = 20, 30
    #start, end = 10, 20
    #start, end = 1, 10

    for i in range(start, end):
    
        case_type = 'AD'
        AD = generate_HOG_array(case_type, i+1)
        print('HOG-AD  #{} components stored.\n\n'.format(i+1))

        case_type = 'CN'
        CN = generate_HOG_array(case_type, i+1)
        print('HOG-CN  #{} components stored.\n\n'.format(i+1))

        case_type = 'MCI'
        MCI = generate_HOG_array(case_type, i+1)
        print('HOG-MCI #{} components stored.\n\n'.format(i+1))
        
        F = [AD, CN, MCI]
        F = np.vstack(F)

        np.save(hog_merged_file.format(i+1), F)
        print('HOG-merged-#{} component saved.'.format(i+1))
        os.system('cls')

    print('All HOG feats saved.')


if __name__ == "__main__":
    start_time = time.time()
    #limit = get_max_slice(r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\{}_iHOG")
    #print(limit) #obtained result 69
    
    #perform_data_generation()

    merge_HOG_array()

    e = int(time.time() - start_time)
    print('\nTime elapsed- {:02d}:{:02d}:{:02d}'.format(e //3600, (e % 3600 // 60), e % 60))
