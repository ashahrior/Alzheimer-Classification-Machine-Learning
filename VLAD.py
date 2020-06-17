import cv2
import itertools
import numpy as np
import FeatureComputation as fc
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt


def VLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

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
        l,h = fc.High_Low(img,i+1)
        img = fc.Change_Range(img,i+1,l,h)

        final_des = list()
        for j in range(low,high):
            cv2.imwrite('photo.jpg',img[j])
            img1 = cv2.imread('photo.jpg',0)
            kp,des = describeORB(img1)

            r = des.shape[0]
            c = des.shape[1]
            row = list()
            if r>=300:
                for k in range(300):
                    for m in range(c):
                        row.append(des[k,m])
            else:
                for k in range(r):
                    for m in range(c):
                        row.append(des[k,m])

                for k in range(300-r):
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
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    return est

def describeORB( image):
    #An efficient alternative to SIFT or SURF
    #doc http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    #ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor
    #with many modifications to enhance the performance
    orb=cv2.ORB_create()
    kp, des=orb.detectAndCompute(image,None) #Image should be .jpeg format
    return kp,des

def all_descriptors(loc,low,high,n,descriptors,group):
    '''
    :param loc: Where the files are
    :param low: lowest number of the slice that would be selected
    :param high: highest number of the slice that would be selected
    :param n: numbers of nifti files in the location
    :return: a list of descriptors ...
    '''
    size = 0
    h_len = 0
    for i in range(n):
        print('{}-Data-{}'.format(group,i+1))
        img = np.load(loc+'data{}.npy'.format(i+1),allow_pickle=True)
        l,h = fc.High_Low(img,i+1)
        img = fc.Change_Range(img,i+1,l,h)

        final_des = list()
        for j in range(low,high):
            cv2.imwrite('photo.jpg',img[j])
            img1 = cv2.imread('photo.jpg',0)
            kp,des = describeORB(img1)

            #print(des.shape)
            '''
            if len(kp) > h_len:
                h_len = len(kp)
                print(h_len)
            '''

            if des.any() != None:
                r = des.shape[0]
                c = des.shape[1]
                #size = size + 300*c
                row = list()
                if r>=300:
                    for k in range(300):
                        for m in range(c):
                            row.append(des[k,m])
                else:
                    for k in range(r):
                        for m in range(c):
                            row.append(des[k,m])

                    for k in range(300-r):
                        for m in range(c):
                            row.append(0)

                row = np.asarray(row)
                final_des.append(row)
            else:
                row = list()
                for k in range(300):
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


'''
##### Main Program #######
img = np.load('Train/AD/data3.npy',allow_pickle=True)
cv2.imwrite('photo1.jpg',img[80])
l,h = fc.High_Low(img,5)
img = fc.Change_Range(img,4,l,h)

img1 = cv2.imread('photo1.jpg',0)

image = img[80]

kp,des = describeORB(img1)


orb = cv2.ORB()
kp = orb.detect(img[80],None)


plt.imshow(des,cmap='gray')
plt.show()
'''


#################### 1. Making Ready for All_features ########

Total = 0

print('#######################')
loc = 'Train/AD/'
n = 20
low = 60
high = 122
Total += n
des = list()
des = all_descriptors(loc,low,high,n,des,'AD')

print('#######################')
loc = 'Test/AD/'
n = 9
Total += n
des = all_descriptors(loc,low,high,n,des,'AD')

print('#######################')
loc = 'Train/CN/'
n = 24
Total += n
des = all_descriptors(loc,low,high,n,des,'CN')

print('#######################')
loc = 'Test/CN/'
n = 10
Total += n
des = all_descriptors(loc,low,high,n,des,'CN')

print('#######################')
loc = 'Train/MCI/'
n = 31
Total += n
des = all_descriptors(loc,low,high,n,des,'MCI')

print('#######################')
loc = 'Test/MCI/'
n = 14
Total += n
des = all_descriptors(loc,low,high,n,des,'MCI')

des = list(itertools.chain.from_iterable(des)) #Flatten
des = np.asarray(des)

np.save('vlad_Feat.npy',des)

#################### 2. Making Visual Words #############
'''
visualDict = kMeansDictionary(des_ad,256)

############# 3. Getting the VLAD descriptors ########

vlad_des = getVLADDescriptors(loc,visualDict,low,high,n)
'''
