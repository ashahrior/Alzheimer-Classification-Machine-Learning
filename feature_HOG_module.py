import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import file_locations_module as flocate 


##### Applying PCA method 
def apply_PCA(feature, no_comp):
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components = no_comp)
    pcomp = pca.fit_transform(X)

    return pcomp


def generate_HOG_array(case_type, number_of_files, target, no_comp, F):
    '''
    :param adrs: 'folder/folder/' where the features are saved
    :param n: number of data files
    :param target: 1 for AD, 2 for CN and 3 for MCI
    :param no_comp: number of PCA
    :param F: Feature Array, just send a list
    :return: returns the updated feature list
    '''
    hog_feat_file_form = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_data\{}_HOG_256x128\256x128_hogFeat{}_data{}.npy"
    
    for i in range(number_of_files):
        hog_data = np.load(hog_feat_file_form.format(case_type, case_type,i+1), allow_pickle=True)
        comp = apply_PCA(hog_data, no_comp)
        row = []
        for j in range(111):
            for k in range(no_comp):
                row.append(comp[j][k])
        row.append(target)
        F.append(row)


def merge_HOG_array(totalComp):
    '''
    :param ad: ADRS of AD data
    :param cn: ADRS of CN data
    :param mci: ADRS of MCI data
    :param totalComp: highest values of PCA
    :return:
    '''
    n_AD_file = 54 #Total datafiles of AD
    n_CN_file = 115
    n_MCI_file = 133

    hog_merged_file = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\HOG_merged\HOG_merged_feat{}.npy'

    for i in range(totalComp):
        F = []
        case_type = 'AD'
        F = generate_HOG_array(case_type, n_AD_file, 1, i+1, F)
        print('HOG-AD feature list for #{} component stored.'.format(i+1))

        case_type = 'CN'
        F = generate_HOG_array(case_type, n_CN_file, 2, i+1, F)
        print('HOG-CN feature list for #{} component stored.'.format(i+1))
        
        case_type = 'MCI'
        F = generate_HOG_array(case_type, n_MCI_file, 3, i+1, F)
        print('HOG-MCI feature list for #{} component stored.'.format(i+1))
        
        np.save(hog_merged_file.format(i+1),F)
        print('HOG merged feature list .npy for #{} component saved.'.format(i+1))

    print('All The HOG Features Arrays saved Successfully.')


################# Main Program ##############

merge_HOG_array(111)
