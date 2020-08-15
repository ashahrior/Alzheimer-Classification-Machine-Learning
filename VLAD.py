import time
import itertools
import os, sys

import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn import model_selection
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


npy_data_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_{}npy\\"

vlad_data_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\\"

orb_data_path = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\{}-ORB\\"


### ORB generation process starts

def add_nan(data, limit):
    """[append nan value to files]

    Args:
        data ([ndarray]): [source data file]

    Returns:
        [ndarray]: [numpy array with nan appended]
    """
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
    interpolated = df.interpolate(method='linear')
    return interpolated.to_numpy()


def perform_data_generation(data, limit):
    nan_data = add_nan(data, limit)
    interpolated_data = interpolate_data(nan_data)
    return interpolated_data


def get_ORBz(data):
    orb = cv2.ORB_create()
    key_points, descriptors = orb.detectAndCompute(data, None)
    return descriptors


def get_file_descriptors(fname, data):
    file_descriptors = np.array([])
    flag = False
    for slice in data:
        limit = 413
        descriptor = get_ORBz(slice)
        #print(fname,x,descriptor.shape, end=' -> ')

        modified_descriptor = perform_data_generation(descriptor, limit)
        slice_desc = modified_descriptor.flatten()

        if flag == False:
            file_descriptors = [slice_desc]
            flag = True
            continue

        file_descriptors = np.concatenate((file_descriptors, [slice_desc]))

    file_descriptors = np.array(file_descriptors)
    print(fname, file_descriptors.shape, end='->')
    return file_descriptors


def get_descriptors(case='AD'):
    os.chdir(npy_data_path.format(case))
    case_descriptors = []
    
    for file in os.listdir():
        fname, fext = os.path.splitext(file)
        data = np.load(file, allow_pickle=True)
        
        file_descriptors = get_file_descriptors(fname, data)
        
        limit = 69
        file_descriptors = perform_data_generation(file_descriptors, limit)
        print(file_descriptors.shape)
        file_descriptors = file_descriptors.flatten()
        
        print(file_descriptors, end=' -')
        case_descriptors.append(file_descriptors)
        print('done\n')
    return case_descriptors


def generate_ORBz():
    cases = ['AD', 'CN', 'MCI']
    for case in cases:
        case_descriptors = get_descriptors(case)
        case_descriptors = np.array(case_descriptors).astype('uint8')
        print(case_descriptors.shape)
        print(case_descriptors)
        save_file = vlad_data_path + f'{case}-ORB.npy'
        np.save(save_file, case_descriptors)
    return


def merge_ORBz():
    ad_orb = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\AD-ORB.npy"
    cn_orb = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\CN-ORB.npy"
    mci_orb = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\MCI-ORB.npy"

    ad_data = np.load(ad_orb, allow_pickle=True)
    print('AD-ORB loaded.')
    cn_data = np.load(cn_orb, allow_pickle=True)
    print('CN-ORB loaded.')
    mci_data = np.load(mci_orb, allow_pickle=True)
    print('MCI-ORB loaded.')

    merged_orb = np.vstack([ad_data, cn_data, mci_data])
    print('ORBz merged.')

    save_file = vlad_data_path + "ORB_ad-cn-mci.npy"
    np.save(save_file, merged_orb)
    print('ORB_ad-cn-mci.npy saved.')

    return

### ORB generation process ends


### Generation of visual dictionary starts

def generate_kMeansDict(training, k):
    '''
    :param training: Descriptors obtained from SIFT,ORB, or something else
    :param k: number of visual words or clusters..
    :return: returns the words
    '''
    #K-means algorithm
    print('Inside kMeansDictionary function.')
    est = KMeans(n_clusters=k, init='k-means++', verbose=1).fit(training)
    print('k-means Dictionary generated.')
    return est


def get_visual_dict(path):
    data = np.load(path, allow_pickle=True)
    print('ORB_ad-cn-mci.npy shape->', data.shape)
    k = 16
    visual_dict = generate_kMeansDict(data, k)
    print("Visual dictionary obtained.")

    model_file = vlad_data_path + f"KMeans_{k}_visual_dict_model.sav"
    pickle.dump(visual_dict, open(model_file, 'wb'))
    print(f'Visual dictionary with {k} clusters model saved.')
    return

### Generation of visual dictionary ends


### Generation of VLAD starts

def make_VLAD(n, descriptors, visual_dict):
    centers = visual_dict.cluster_centers_
    labels = visual_dict.labels_
    k = visual_dict.n_clusters

    centroid = centers[labels[n]]
    centroid = centroid.reshape((69, 13216))

    V = np.sum(descriptors - centroid, axis=0)
    
    #m, d = descriptors.shape
    #V = np.zeros([k,d])
    '''
    for i in range(k):
        if np.sum(predictedLabels == i) > 0:
            V[i] = np.sum(descriptors[predictedLabels == i, :]- centers[i], axis=0)
    print(V.shape)
    '''
    #V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))
    print(V)
    # L2 normalization
    V = V/np.sqrt(np.dot(V, V))
    print(V)
    print()
    return V

def append_target(data):
    ad = np.full((54,), 1)
    cn = np.full((54,), 2)
    mci = np.full((54,), 3)
    target = np.hstack([ad, cn, mci])
    final_data = np.column_stack((data,target))
    return final_data

def get_VLAD():
    k = 16
    visual_dict = pickle.load(open(vlad_data_path+f"KMeans_{k}_visual_dict_model.sav", "rb"))

    orb_all = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\ORB_ad-cn-mci.npy"

    orb_data = np.load(orb_all, allow_pickle=True)
    vlad_desc = []
    n = 0
    for row in orb_data:
        data = row.reshape((69, 13216))
        result = make_VLAD(n, data, visual_dict)
        vlad_desc.append(result)
        n += 1
    vlad_desc = np.array(vlad_desc)
    print(vlad_desc.shape)
    vlad_desc_target = append_target(vlad_desc)
    np.save(vlad_data_path + f"VLAD_{k}_feat.npy", vlad_desc_target)
    print("VLAD saved.")
    '''
    for line in orb_data:
        vlad_line = make_VLAD(line, visual_dict)
        vlad_orb.append(vlad_line)
        break
    '''
    '''
    for case in cases:
        descriptors = orb_data_path.format(case)
        case_vlad = make_VLAD(descriptors, visual_dict)
        save_as = vlad_data_path.format(case) + f"{case}-VLAD.npy"
        np.save(save_as, case_vlad)
        print(case, 'VLAD saved.\n')
        return
    '''

    '''
    for case in cases:
        os.chdir(npy_data_path.format(case))
        file_counter = 1
        case_vlad = []
        for file in os.listdir():
            fname, fext = os.path.splitext(file)
            data = np.load(file, allow_pickle=True)
            print(file,'loaded',end=' - ')
            file_descriptors = get_file_descriptors(fname, data)
            print(data.shape,'descriptor gained',end=' - ')
            limit = 69
            file_descriptors = perform_data_generation(file_descriptors, limit)

            save_as = orb_data_path.format(case)+f"{fname}_orb.npy"
            np.save(save_as, file_descriptors)
            print('saved.')

            file_vlad = make_VLAD(file_descriptors, visual_dict)
            case_vlad.append(file_vlad)
            print(fname,'VLAD received.\n')
        
        case_vlad = np.asarray(case_vlad)
        save_as = vlad_data_path.format(case) + f"{case}-VLAD.npy"
        np.save(save_as, case_vlad)
        print(case, 'VLAD saved.\n')
    '''    
    return

### Generation of VLAD ends


if __name__ == "__main__":
    
    start_time = time.time()
    
    # step-1
    #generate_ORBz()

    #merge_ORBz()

    # step-2
    #get_visual_dict(r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\CLAHE_NPY\CLAHE_VLAD\ORB_ad-cn-mci.npy")

    # step-3
    #cases = ['AD', 'CN', 'MCI']
    get_VLAD()
    
    e = int(time.time() - start_time)
    print('Time elapsed- {:02d}:{:02d}:{:02d}'.format(e //3600, (e % 3600 // 60), e % 60))

# [(413, 326), (410, 359), (411, 122)]
