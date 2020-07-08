import numpy as np, os, pandas

fol = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_Imputed_idata\imputed_i{}\\"

fsrc = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_Imputed_idata\imputed_i{}\\{}-{}1-imp.npy"

cases = ['AD', 'CN', 'MCI']

case = 'AD'

feats = ['asm', 'contrast', 'correlation',
         'dissimlarity', 'energy', 'homogeneity']

form = "{}-{}1-imp.npy"


