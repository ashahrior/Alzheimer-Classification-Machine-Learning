import os, numpy as np
from numpy import nan
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

fol = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_idata\\i{}"

tar = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_Imputed_idata\imputed_i{}\\"

#src = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_idata\iMCI\MCI-contrast76.npy"

#src = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\Normalized_NPY\AD_normNPY\{}-data1.npy"

lim = 69


def add_nan(data):
    slices, cols = data.shape
    print(slices)
    zero_pad = np.zeros((lim-slices,4))
    padded_data = np.vstack([data, zero_pad])
    nan_data = np.where(padded_data == 0, np.nan, padded_data)
    return nan_data


def interpolate_data(data):
    df = pd.DataFrame(data)
    return df.interpolate(method='spline', order=1)


def perform_data_generation(cases=['AD', 'CN', 'MCI']):
    for c in cases:
        os.chdir(fol.format(c))
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
            np.save(tar.format(c) + f"{fname}-imp{fext}", interpolated_data)

            print(f"{fname}-imp{fext} saved.")
            count += 1

        print(f'{c} - {count} done')


def check_data(cases=['AD', 'CN', 'MCI']):
    count = 0
    for c in cases:
        os.chdir(tar.format(c))
        print(f'Inside {c} dir')
        for file in os.listdir():
            data = np.load(file, allow_pickle=True)
            if data.shape[0] != lim:
                print(file, data.shape)
                count += 1
    if count != 0:
        print(f'{count} troubled files')
    return

if __name__ == "__main__":
    cases = ['AD', 'CN', 'MCI']
    #perform_data_generation()
    #check_data()


