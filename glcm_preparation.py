import numpy as np
from skimage.feature import greycomatrix, greycoprops
import os, re

norm_data = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\Normalized_NPY\{}_normNPY\\"

glcm_feats = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\GLCM_idata\i{}\\"



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





if __name__ == "__main__":
    pass
