import time
import itertools
import sys

import cv2
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from functional_modules import feature_computation_module as fc

def beeper():
    for i in range(5):
        sys.stdout.write('\a')
        time.sleep(2)


def VLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V,V))
    return V


def getVLADDescriptors(path,visualDictionary,low,high,n):
    descriptors=list()
    idImage =list()
    for i in range(n):
        print('Data-{}'.format(i+1))
        img = np.load(path + 'data{}.npy'.format(i+1), allow_pickle=True)
        l, h = fc.get_high_low_gray_level(img, i+1)
        img = fc.change_image_dynamic_range(img, i+1, l, h)

        final_des = list()
        for j in range(low,high):
            cv2.imwrite('photo.jpg',img[j])
            img1 = cv2.imread('photo.jpg',0)
            kp, des = describeORB(img1)
            
            if des is not None:
                r = des.shape[0]
                c = des.shape[1]
                row = list()
                if r>=50:
                    for k in range(50):
                        for m in range(c):
                            row.append(des[k,m])
                else:
                    for k in range(r):
                        for m in range(c):
                            row.append(des[k,m])

                    for k in range(50-r):
                        for m in range(c):
                            row.append(0)

            row = np.asarray(row)
            final_des.append(row)
        else:
            row = list()

            for k in range(50):
                    for m in range(32):
                        row.append(0)

            row = np.asarray(row)
            final_des.append(row)

        final_des = np.asarray(final_des)
        print('des calculated..')

        print('VLAD-method called ..')
        v=VLAD(final_des,visualDictionary)
        print('VLAD recieved...')
        descriptors.append(v)
        idImage.append(i)

    #list to array
    descriptors = np.asarray(descriptors)
    return descriptors


def  kMeansDictionary(training, k):
    '''
    :param training: Descriptors obtained from SIFT,ORB, or something else
    :param k: number of visual words or clusters..
    :return: returns the words
    '''
    #K-means algorithm
    print('Inside kMeansDictionary function.')
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    print('Exiting kMeansDictionary')
    return est


def describeORB( image):
    #An efficient alternative to SIFT or SURF
    #doc http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    #ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor
    #with many modifications to enhance the performance
    orb=cv2.ORB_create()
    kp, des=orb.detectAndCompute(image, None) #Image should be .jpeg format
    return kp,des


def all_descriptors(loc, low, high, n, descriptors, group):
    '''
    :param loc: Where the files are
    :param low: lowest number of the slice that would be selected - 40
    :param high: highest number of the slice that would be selected - 150+1
    :param n: numbers of files in the location
    :return: a list of descriptors ...
    '''
    size = 0
    h_len = 0
    for i in range(n):
        print('{}-Data-{}'.format(group, i+1))
        img = np.load(loc+'data{}.npy'.format(i+1), allow_pickle=True)
        l, h = fc.get_high_low_gray_level(img, i+1)
        img = fc.change_image_dynamic_range(img, i+1, l, h)

        final_des = list()
        for j in range(low, high):
            cv2.imwrite('photo.jpg', img[j])
            img1 = cv2.imread('photo.jpg', 0)
            kp,des = describeORB(img1)
            
            #print(des.shape)
            '''
            if len(kp) > h_len:
                h_len = len(kp)
                print(h_len)
            '''

            if des is not None:
                r = des.shape[0]
                c = des.shape[1]
                #size = size + 50*c
                row = list()
                if r>=50:
                    for k in range(50):
                        for m in range(c):
                            row.append(des[k,m])
                else:
                    for k in range(r):
                        for m in range(c):
                            row.append(des[k,m])

                    for k in range(50-r):
                        for m in range(c):
                            row.append(0)

                row = np.asarray(row)
                final_des.append(row)
            else:
                row = list()
                for k in range(50):
                    for m in range(32):
                        row.append(0)

                row = np.asarray(row)
                final_des.append(row)

        final_des = np.asarray(final_des)
        descriptors.append(final_des)

        #print('Total Size of Descriptors: {} MB'.format(size/128318))
        #c = input('Enter for next: ')

    #descriptors = list(itertools.chain.from_iterable(descriptors)) #Flatten
    #descriptors = np.asarray(descriptors)
    #print(h_len)
    return descriptors


#################### 1. Making Ready for All_features ########
def get_all_descriptors(low, high):
    des = list()
    total = 0
    print('#######################')
    loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\AD_mainNPY\\'
    n = 54
    total += n
    des = all_descriptors(loc, low, high, n, des, 'AD')
    #beeper()
    #input('AD complete. Enter to continue >>')


    print('#######################')
    loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\CN_mainNPY\\'
    n = 54#115
    total += n
    des = all_descriptors(loc, low, high, n, des, 'CN')
    #beeper()
    #input('CN complete. Enter to continue >>')


    print('#######################')
    loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\MCI_mainNPY\\'
    n = 54#133
    total += n
    des = all_descriptors(loc, low, high, n, des, 'MCI')
    #beeper()
    #input('MCI complete. Enter to continue >>')


    des = list(itertools.chain.from_iterable(des)) #Flatten
    des = np.asarray(des).astype('uint8')

    np.save('E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\\VLAD_50feat.npy', des)

    print()
    return


############# 2. Making Visual Words #############
def get_visual_dict():
    vlad_data_file = np.load(
        r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\VLAD_50feat.npy", allow_pickle=True)
    print(vlad_data_file.shape)
    print(np.max(vlad_data_file))
    print(vlad_data_file.dtype)
    print(vlad_data_file[0, -1])
    visualDict = kMeansDictionary(vlad_data_file, 256)
    print('Visual Dictionary obtained.')

    model_file = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\KMean50_model.sav'
    pickle.dump(visualDict, open(model_file, 'wb'))
    print('Visual Dictionary model saved.')
    return


############# 3. Getting the VLAD descriptors #############
def get_vlad_desc(low, high):
    n = 54

    ad_loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\AD_mainNPY\\'
    cn_loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\CN_mainNPY\\'
    mci_loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\MCI_mainNPY\\'

    visualDict = pickle.load(open("E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\KMean50_model.sav", 'rb'))

    '''
    vlad_ad = getVLADDescriptors(ad_loc, visualDict, low, high, n)

    np.save('E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\\vlad50_ad.npy', vlad_ad)

    print()
    input('AD complete. Enter to continue >>')
    '''

    
    '''
    vlad_cn = getVLADDescriptors(cn_loc, visualDict, low, high, n)

    np.save('E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\\vlad50_cn.npy', vlad_cn)

    print()
    input('CN complete. Enter to continue >>')
    '''

    ''''''
    vlad_mci = getVLADDescriptors(mci_loc, visualDict, low, high, n)

    np.save('E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\\vlad50_mci.npy', vlad_mci)

    print()
    input('MCI complete. Enter to continue >>')
    ''''''
    return

if __name__ == "__main__":
    start_time = time.time()
    low = 40
    high = 151
    #get_all_descriptors(low, high)
    #get_visual_dict()
    get_vlad_desc(low,high)
    e = int(time.time() - start_time)
    print('Time elapsed- {:02d}:{:02d}:{:02d}'.format(e //3600, (e % 3600 // 60), e % 60))
