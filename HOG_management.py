import os
import time
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import skimage
from skimage import exposure
from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize

from functional_modules import data_viewer_module as dv


cases = ["AD", "CN", "MCI"]
#cases = ["AD"]

resizeRow, resizeCol = 256, 128
resolution = f"{resizeRow}x{resizeCol}"
title = r"{}-{}-CLAHE-HOG-{}"

hog_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\\"

hog_feat_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\{}_claheHOG\\"

big_hogs_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_HOG\Big_HOGs\\"

#hog_img_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\{}_HOG_img_"+resolution+"\\"

norm_npy_fold = "E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_{}npy\\"

#imp_hog_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\HOG_idata\imputed_HOG\\"


hog


limit = 69


def calculate_HOG_feats():
    global resolution
    for cs in cases:
        os.chdir(norm_npy_fold.format(cs))
        for f in os.listdir():
            fname, fext = os.path.splitext(f)
            serial = re.findall('\d+', fname)[0]
            hog_file_name = title.format(cs, serial, resolution, fext)
            print(f'\n{f} opened')
            data = np.load(f, allow_pickle=True)
            hog_feats = []
            hog_images = []
            for slices in range(data.shape[0]):
                imageSlice = data[slices]

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
                print('HOG gained')
                hog_feats.append(fd)
                #hog_images.append(hog_img)

            hog_feats = np.asarray(hog_feats)
            #hog_images = np.asarray(hog_images)
            #dv.Show(hog_images)
            #break
            np.save(hog_feat_fold.format(cs) + hog_file_name, hog_feats)
            print(f'{hog_file_name} hog feat saved.')
        #break
    return


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
            
            #case_folder = imp_hog_fold + f"{c}_impHOG\\"
            case_folder = hog_feat_fold.format(c)
            
            #np.save(case_folder + f"{fname}-imp{fext}", interpolated_data)
            np.save(case_folder+file, interpolated_data)

            #print(f"{fname}-imp{fext} saved.")
            print(f'Interpolated {file} saved.')
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


def append_HOGs(src, targ, case):

    target = {
        'AD': 1, 'CN': 2, 'MCI': 3
    }
    os.chdir(src)
    big = np.array([])
    for file in os.listdir():
        print(file, end=' - ')
        data = np.load(file, allow_pickle=True)
        data = data.flatten()
        data = np.append(data, target[case])
        big = data
        break
    counter = 0
    size = (big.size * big.itemsize) / 1024 / 1024
    print(size)
    for file in os.listdir():
        if counter == 0:
            counter = 1
            continue
        print(file, end=' - ')
        data = np.load(file, allow_pickle=True)
        data = data.flatten()
        data = np.append(data, target[case])
        big = np.vstack((big, data))
        size = (big.size * big.itemsize) / 1024 / 1024
        print(size)
    print('Append done', ((big.size * big.itemsize) / 1024 / 1024))
    file = targ + f'{case}-HOG.npy'
    np.save(file, big)
    return


def merge_HOGs(src, targ):

    os.chdir(src)
    big = np.array([])

    ll = []

    for file in os.listdir():
        print(file, end=' - ')
        data = np.load(file)
        size = (data.size * data.itemsize) / 1024 / 1024
        print(size, end=' - ')
        ll.append(data)
        print('appended')
    big = np.vstack(ll)
    np.save('CLAHE-HOG-MERGED.npy', big)
    print('saved')
    return


if __name__ == "__main__":
    start_time = time.time()
    
    # step-1
    #calculate_HOG_feats()

    # step-2
    #limit = get_max_slice(hog_feat_fold)
    #print('\n\n',limit) #obtained result 69
    
    # step-3
    #perform_data_generation()

    # step-4
    #case = 'MCI'
    #append_data(hog_feat_fold.format(case), big_hogs_fold, case)

    # step-5
    #src = targ = big_hogs_fold
    #merge_HOGs(src,targ)

    e = int(time.time() - start_time)
    print('\nTime elapsed- {:02d}:{:02d}:{:02d}'.format(e //3600, (e % 3600 // 60), e % 60))
