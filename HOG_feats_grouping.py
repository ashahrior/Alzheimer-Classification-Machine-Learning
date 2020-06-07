import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os

from functional_modules import file_locations_module as flocate


##### Applying PCA method 
def apply_PCA(feature, no_comp):
    print('Applying PCA for #{} components'.format(no_comp))
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components = no_comp)
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
    print('Inside generate_HOG_array() function with case type-{} for number of components-{}'.format(case_type,no_comp))
    print()

    for i in range(number_of_files):
        hog_data = np.load(hog_feat_file_form.format(case_type, case_type, i+1), allow_pickle=True)
        print('HOG data for',hog_feat_file_form.format(case_type, case_type, i+1),'loaded.')
        
        comp = apply_PCA(hog_data, no_comp)
        print('PCA applied for {} case in file #{} for {} components.'.format(case_type, i+1, no_comp))
        
        row = []
        for j in range(111):
            for k in range(no_comp):
                row.append(comp[j][k])
        row.append(target)

        F.append(row)
        print('Row for case-%s for file #%d with %d components appended.'%(case_type,i+1,no_comp))
        print()
    return F


def merge_HOG_array(totalComp):
    '''
    :param totalComp: highest values of PCA
    :return:
    '''
    n_AD_file = 54 #Total datafiles of AD
    n_CN_file = 115
    n_MCI_file = 133

    hog_merged_file = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_merged\HOG_merged_feat{}.npy'
    
    
    start, end = 100, totalComp
    
    #start, end = 90, 100
    start, end = 80, 90
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
        
        np.save(hog_merged_file.format(i+1),F)
        print('HOG merged feature list .npy for #{} component saved.'.format(i+1))
        print()
        print()
        os.system('cls')

    print('All The HOG Features Arrays saved Successfully.')


################# Main Program ##############
if __name__ == "__main__":
    merge_HOG_array(111)
