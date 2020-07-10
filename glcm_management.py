import os
import re

import numpy as np
import pandas as pd 
from numpy import nan
from skimage.feature import greycomatrix, greycoprops
from sklearn.impute import SimpleImputer


norm_data = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\Normalized_NPY\{}_normNPY\\"

glcm_feats = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_idata\i{}\\"

imputed_data_fold = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_Imputed_idata\imputed_i{}\\"

intrst_data_fol = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\\"


limit = 0

feats = ['asm', 'contrast', 'correlation',
         'dissimlarity', 'energy', 'homogeneity']

cases = {
    'AD': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
           29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'CN': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 22, 24, 25, 28, 29, 30, 33, 39, 40, 41, 49, 59, 60, 63, 64, 70, 71, 72, 73, 74, 78, 79, 80, 81, 82, 84, 85, 87, 99, 100, 101, 106, 109, 110, 111, 112, 115],

    'MCI': [1, 6, 7, 8, 9, 10, 11, 27, 29, 30, 31, 32, 33, 34, 36, 40, 43, 44, 45, 46, 52, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 65, 66, 67, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 96, 98, 99, 113, 114]

}

case_data = []


def save_glcm_feats(case, serial, features):
    # con, diss, homo, en, corr, asms
    con = features[0]
    diss = features[1]
    homo = features[2]
    en = features[3]
    corr = features[4]
    asms = features[5]

    np.save((glcm_feats.format(case) + "{}-asm{}".format(case, serial)), asms)
    print(f'{case}-{serial} asm', 'saved')

    np.save((glcm_feats.format(case) + "{}-contrast{}".format(case, serial)), con)
    print(f'{case}-{serial} contrast', 'saved')

    np.save((glcm_feats.format(case) + "{}-correlation{}".format(case, serial)), corr)
    print(f'{case}-{serial} correlation', 'saved')

    np.save((glcm_feats.format(case) + "{}-dissimlarity{}".format(case, serial)), diss)
    print(f'{case}-{serial} dissimilarity', 'saved')

    np.save((glcm_feats.format(case) + "{}-energy{}".format(case, serial)), en)
    print(f'{case}-{serial} energy', 'saved')

    np.save((glcm_feats.format(case) + "{}-homogeneity{}".format(case, serial)), homo)
    print(f'{case}-{serial} homogeneity', 'saved')
    return


def get_glcm(case, serial, data):
    
    def get_contrast_feature(matrix_coocurrence):
	    return greycoprops(matrix_coocurrence, 'contrast')


    def get_dissimilarity_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'dissimilarity')


    def get_homogeneity_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'homogeneity')


    def get_energy_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'energy')


    def get_correlation_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'correlation')


    def get_asm_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'ASM')


    con = []
    diss = []
    homo = []
    en = []
    corr = []
    asms = []
    for i in range(data.shape[0]):
        matrix_coocurrence = greycomatrix(data[i], [1], [
                                          0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, normed=False, symmetric=False)

        asm = get_asm_feature(matrix_coocurrence)
        asms.append(asm.flatten())

        contrast = get_contrast_feature(matrix_coocurrence)
        con.append(contrast.flatten())

        correlation = get_correlation_feature(matrix_coocurrence)
        corr.append(correlation.flatten())

        dissimilarity = get_dissimilarity_feature(matrix_coocurrence)
        diss.append(dissimilarity.flatten())

        energy = get_energy_feature(matrix_coocurrence)
        en.append(energy.flatten())

        homogeneity = get_homogeneity_feature(matrix_coocurrence)
        homo.append(homogeneity.flatten())

    features = [con, diss, homo, en, corr, asms]

    save_glcm_feats(case, serial, features)
    return


def calc_glcm(case):
    os.chdir(norm_data.format(case))
    for file in os.listdir():
        data = np.load(file, allow_pickle=True)
        print(file, ' -> ', data.shape)
        serial = re.findall('\d+', file)[0]
        get_glcm(case, serial, data)


def get_max_slice(src):
    """get the maxmimum number of slices of any file for overall dataset

    Args:
        src ([str]): [source location of the data files]
    """
    cases = ['AD', 'CN', 'MCI']
    global limit
    for c in cases:
        os.chdir(src.format(c))
        for file in os.listdir():
            data = np.load(file)
            slices = data.shape[0]
            limit = max(slices, limit)
    return limit


def add_nan(data, n_zeros=4):
    """[append nan value to files]

    Args:
        data ([ndarray]): [source data file]

    Returns:
        [ndarray]: [numpy array with nan appended]
    """
    global limit
    slices = data.shape[0]
    print(slices)
    zero_pad = np.zeros((limit-slices, n_zeros))
    padded_data = np.vstack([data, zero_pad])
    nan_data = np.where(padded_data == 0, np.nan, padded_data)
    return nan_data


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
        os.chdir(glcm_feats.format(c))
        count = 0
        for file in os.listdir():
            print(file, '->', end='')
            fname, fext = os.path.splitext(file)
            data = np.load(file, allow_pickle=True)

            nan_data = add_nan(data)
            print(f'{fname} nan addition done.')

            interpolated_df = interpolate_data(nan_data)
            print(f'{fname} interpolation done.')

            interpolated_data = interpolated_df.to_numpy()
            np.save(imputed_data_fold.format(c) +
                    f"{fname}-imp{fext}", interpolated_data)

            print(f"{fname}-imp{fext} saved.")
            count += 1

        print(f'{c} - {count} done')


def check_data(cases=['AD', 'CN', 'MCI']):
    global limit
    count = 0
    for c in cases:
        os.chdir(imputed_data_fold.format(c))
        print(f'Inside {c} dir')
        for file in os.listdir():
            data = np.load(file, allow_pickle=True)
            if data.shape[0] != limit:
                print(file, data.shape)
                count += 1
    if count != 0:
        print(f'{count} troubled files')
    return


def merge_glcm():
    """merge all the glcm feats of all three cases into a single file in 2D
    """
    for case in ['AD', 'CN', 'MCI']:
        if case == 'AD':
            target = 1
        elif case == 'CN':
            target = 2
        else:
            target = 3
        for serial in cases[case]:
            d = []
            print(f'{case} #{serial} file - ', end='')
            for feat in feats:
                print(f'{feat} -', end=' ')
                form = f"{case}-{feat}{serial}-imp.npy"
                data = np.load(imputed_data_fold.format(case) + form, allow_pickle=True)
                data = data.flatten()
                d.append(data)
            file_serial_data = np.concatenate(d)
            file_serial_data = np.append(file_serial_data, [target])

            #print(file_serial_data)
            #print(file_serial_data.shape)
            case_data.append(file_serial_data)
            print(' done ')
        print(f'{case} done')

    case_data_array = np.array(case_data)
    print(case_data_array.shape)
    np.save(intrst_data_fol+"all_clean_glcm_54.npy", case_data_array)



if __name__ == "__main__":
    pass
