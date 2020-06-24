import time
import itertools

import cv2
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from functional_modules import feature_computation_module as fc


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
        print('Data-{}'.format(i))
        img=np.load(path+'data{}.npy'.format(i+1),allow_pickle=True)
        l, h = fc.get_high_low_gray_level(img, i+1)
        img = fc.change_image_dynamic_range(img, i+1, l, h)

        final_des = list()
        for j in range(low,high):
            cv2.imwrite('photo.jpg',img[j])
            img1 = cv2.imread('photo.jpg',0)
            kp,des = describeORB(img1)

            r = des.shape[0]
            c = des.shape[1]
            row = list()
            if r>=200:
                for k in range(200):
                    for m in range(c):
                        row.append(des[k,m])
            else:
                for k in range(r):
                    for m in range(c):
                        row.append(des[k,m])

                for k in range(200-r):
                    for m in range(c):
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
    return descriptors, idImage

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
                #size = size + 200*c
                row = list()
                if r>=200:
                    for k in range(200):
                        for m in range(c):
                            row.append(des[k,m])
                else:
                    for k in range(r):
                        for m in range(c):
                            row.append(des[k,m])

                    for k in range(200-r):
                        for m in range(c):
                            row.append(0)

                row = np.asarray(row)
                final_des.append(row)
            else:
                row = list()
                for k in range(200):
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

Total = 0
low = 40
high = 151

'''
des = list()

print('#######################')
loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\AD_mainNPY\\'
n = 54
Total += n
des = all_descriptors(loc, low, high, n, des, 'AD')
#input('AD complete. Enter to continue >>')

print('#######################')
loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\CN_mainNPY\\'
n = 54#115
Total += n
des = all_descriptors(loc, low, high, n, des, 'CN')
#input('CN complete. Enter to continue >>')

print('#######################')
loc = 'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\MCI_mainNPY\\'
n = 54#133
Total += n
des = all_descriptors(loc, low, high, n, des, 'MCI')
#input('MCI complete. Enter to continue >>')

des = list(itertools.chain.from_iterable(des)) #Flatten
des = np.asarray(des)

np.save('E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\\VLAD_feat.npy', des)

print()
'''

start_time = time.time()

############# 2. Making Visual Words #############
vlad_data_file = np.load(r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\VLAD_feat.npy", allow_pickle=True)
visualDict = kMeansDictionary(vlad_data_file, 256)
print('Visual Dictionary obtained.')

model_file = r'E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\KMean_model.sav'
pickle.dump(visualDict,open(model_file,'wb'))
print('Visual Dictionary saved.')

'''
############# 3. Getting the VLAD descriptors #############
n = 54
vlad_ad = getVLADDescriptors(ad_loc, visualDict, low, high, n)
vlad_cn = getVLADDescriptors(cn_loc, visualDict, low, high, n)
vlad_mci = getVLADDescriptors(mci_loc, visualDict, low, high, n)
'''

e = int(time.time() - start_time)
print('Time elapsed- {:02d}:{:02d}:{:02d}'.format(e //3600, (e % 3600 // 60), e % 60))

''''''
