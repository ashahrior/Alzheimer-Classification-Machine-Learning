import numpy as np, os, pandas

cases = {
    'AD' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'CN' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 22, 24, 25, 28, 29, 30, 33, 39, 40, 41, 49, 59, 60, 63, 64, 70, 71, 72, 73, 74, 78, 79, 80, 81, 82, 84, 85, 87, 99, 100, 101, 106, 109, 110, 111, 112, 115],

    'MCI' : [1, 6, 7, 8, 9, 10, 11, 27, 29, 30, 31, 32, 33, 34, 36, 40, 43, 44, 45, 46, 52, 55, 56, 57, 58, 59, 60,
           61, 62, 63, 65, 66, 67, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 96, 98, 99, 113, 114]

}


fol = "E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_Imputed_idata\imputed_i{}\\"

tsrc = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\\"

feats = ['asm', 'contrast', 'correlation', 'dissimlarity', 'energy', 'homogeneity']

case_data = []


for case in ['AD', 'CN', 'MCI']:
    if case == 'AD':
        target = 1
    elif case == 'CN':
        target = 2
    else: target = 3
    for serial in cases[case]:
        d = []
        print(f'{case} #{serial} file - ',end='')
        for feat in feats:
            print(f'{feat} -',end=' ')
            form = f"{case}-{feat}{serial}-imp.npy"
            data = np.load(fol.format(case) + form, allow_pickle=True)
            data = data.flatten()
            d.append(data)
        file_serial_data = np.concatenate(d)
        file_serial_data = np.append(file_serial_data,[target])

        #print(file_serial_data)
        #print(file_serial_data.shape)
        case_data.append(file_serial_data)
        print(' done ')
    print(f'{case} done')

case_data_array = np.array(case_data)
print(case_data_array.shape)
np.save(tsrc+"all_clean_glcm_54.npy", case_data_array)


