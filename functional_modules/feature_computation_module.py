import math
import os

import cv2
import nibabel as nib
import numpy as np


####### Loading Data ######
def convert_nii_to_npy(address,location,title=''):
    '''
    :param address: Where the nifti file is.
    :param location: Where to save the file .npy
    :param number: is 0 (zero)
    :return: Saves the .npy files
    '''
    n = 1
    for f in os.listdir(address):
        try:
            print(n,' ',f)
            path = os.path.join(address,f)
            img = nib.load(path)
            img = img.get_data()
            np.save(location+title+'\\Data{}.npy'.format(n),img) ##Check The location
            n += 1
        except Exception as e:
            pass
    print('Image loaded and saved!!!!')

#### Opening npy Data from the file ...
def open_NPY(address,n):
    data = np.load(address.format(n),allow_pickle=True)
    #print('.npy file opened and returned')
    return data

#### Opening npy Data from the file ...
def open_interest_data(address,n):
    data = np.load(address.format(n),allow_pickle=True)
    print('.npy file opened and returned')
    return data[40:151]


############# Return Low and High Gray Level###
def get_high_low_gray_level(img):
    low = 1000
    high = 0
    for i in range(img.shape[0]):
        max = np.amax(img[i,:,:])
        min = np.amin(img[i,:,:])

        if low>min:
            low = min
        if high < max:
            high = max
    return low,high

####### Changing the Dynamic range of the Image ...
def change_image_dynamic_range(img,low,high):
    newImg = 255*((img - low)/(high - low))
    return newImg

##### Converting into INtegers
def convert_into_integer(img,n):
    Nimg = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                Nimg[i,j,k] = int(img[i,j,k])
        print('Converting Integer Slice - {} - {}'.format(n,i))
    print('Integer Image Returned')
    return Nimg

##### Calculate glcm of an mri data....
def get_GLCM(img,n):
    glcm = np.zeros((img.shape[0],260,260)) ### values from 300 to 600, considering as 0 to 300 by subtracting 300.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]-1):
                #print(img[i,j,k],img[i,j,k+1])
                #kgg = input('Hello : ')
                glcm[i,int(img[i,j,k]),int(img[i,j,k+1])] += 1
        print('Computing GLCM slice {}-{}'.format(n,i))
    print('GLCM returned')
    return glcm

###### Normalizing The Inputs ...
def normalize_GLCM(img,n):
    sum = 0
    for i in range(img.shape[0]):
        sum = sum + np.sum(img[i,:,:])
    Nimg = img/sum
    print('Normalized GLCM data-{} returned'.format(n))
    return Nimg

##### Computing Homogeneity ....
def get_homogeneity(img,n):
    C = []
    for i in range(img.shape[0]):
        con = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                con = con + img[i,j,k]/(1+abs(j-k))
                #print('{}-{}: {}'.format(j,k,entropy))
        C.append(con)
        print('slice {}-{}'.format(n,i),'Homogeneity: {}'.format(con))
    print('Homogeneity returned')
    return C

##### Computing Disimilarity ....
def get_dissimilarity(img,n):
    C = []
    for i in range(img.shape[0]):
        con = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                con = con + abs(j-k)*img[i,j,k]
                #print('{}-{}: {}'.format(j,k,entropy))
        C.append(con)
        print('slice {}-{}'.format(n,i),'Dissimilarity: {}'.format(con))
    print('Dissimilarity returned')
    return C

##### Computing ASM = Anglar Second Moment ....
def get_ASM(img,n):
    C = []
    for i in range(img.shape[0]):
        con = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                con = con + (img[i,j,k]**2)
                #print('{}-{}: {}'.format(j,k,entropy))
        C.append(con)
        print('slice {}-{}'.format(n,i),'ASM: {}'.format(con))
    print('ASM returned')
    return C

##### Computing IDM = Inverse Difference Moment ....
def get_IDM(img,n):
    C = []
    for i in range(img.shape[0]):
        con = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                con = con + (img[i,j,k]/(1+((j-k)**2)))
                #print('{}-{}: {}'.format(j,k,entropy))
        C.append(con)
        print('slice {}-{}'.format(n,i),'IDM: {}'.format(con))
    print('IDM returned')
    return C

##### Computing Contrast ....
def get_contrast(img,n):
    C = []
    for i in range(img.shape[0]):
        con = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                c = ((j-k)**2)*img[i,j,k]
                con = con+c
                #print('{}-{}: {}'.format(j,k,entropy))
        C.append(con)
        print('slice {}-{}'.format(n,i),'Contrast: {}'.format(con))
    print('Contrast returned')
    return C

############### Computing Entropy from Image File....
def get_entropy(img,n):
    E = []
    for i in range(img.shape[0]):
        entropy = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i,j,k] <= 255 and img[i,j,k]>0:
                    e = - (img[i,j,k]/255)*(np.log(img[i,j,k]/255)/np.log(2))
                    entropy = entropy+e
                    #print('{}-{}: {}'.format(j,k,entropy))
        E.append(entropy)
        print('slice {}-{}'.format(n,i),'Entropy: {}'.format(entropy))

    return E
    #print(entropy)

### Computing Brightness  of slices of gray scale images from 3d npy data ...
def get_brightness(img,n):
    ### Calculating brightness with a threshold value...
    brightness = []
    N = img.shape[1]*img.shape[2]
    for i in range(img.shape[0]):
        total  = np.sum(img[i])
        brightness.append(total/N)
        print('Slice {}-{} brightness : {}'.format(n,i,brightness[i]))
    print('Brightness returned')
    return brightness

### Computing Variances  of slices of gray scale images from 3d npy data ...
def get_variance(img,mean,n):
    ### Calculating brightness with a threshold value...
    variance = []
    N = img.shape[1]*img.shape[2]
    for i in range(img.shape[0]):
        total  = np.sum((img[i]-mean[i])**2)
        variance.append(total/(N-1))

        print('Slice {}-{} Variance : {}'.format(n,i,variance[i]))
    print('Variance returned')
    return variance

##### Loading All The features into One Array ....
def load_features(address,n):
    Feature = []
    for i in range(n):
        feature = np.load(address.format(i+1),allow_pickle=True)
        Feature.append(feature)
    return Feature


######### getting distance from the centroids .....
def get_centroidal_distance(c1,c2,c3,data):
    dist1 = abs(c1 - data)
    dist2 = abs(c2 - data)
    dist3 = abs(c3 - data)
    return dist1,dist2,dist3

#### Normalizing Features into o and 100 ...
def normalize_features(feature):
    m = np.max(feature)
    l = np.min(feature)

    return 100*((feature-l)/(m-l))


############ Thresholding between low to high ....
def thresholding_skull(img,n,low,high):
    newImg = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i,j,k] >low and img[i,j,k] < high:
                    newImg[i,j,k] = img[i,j,k]
                else:
                    newImg[i,j,k] = 0
        print('slice {}-{}'.format(n,i))
    return newImg

### Computing Brightness  of whole 3d npy data ...
def compute_3D_brightness(img,n):
    ### Calculating brightness with a threshold value...
    brightness = 0
    count_total = 0
    for i in range(img.shape[0]):
        total = 0
        count = 0
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i,j,k] >300 and img[i,j,k] < 600:
                    total = total + img[i,j,k]
                    count = count +1
        if count != 0 :
            brightness = brightness + (total/count)
            count_total = count_total +1

        print('slice {}-{}'.format(n,i))
    if count_total != 0 :
        brightness = brightness/count_total

    return brightness

#### Return the brightnesses ...
'''def get_brightness(A):
    train_Data = np.load(A,allow_pickle=True)
    print(train_Data)
    #return  train_Data
'''

###### Normalizing The Inputs ...
def normalize(img,n):
    low,high = get_high_low_gray_level(img)
    Nimg = img/high
    print('Data-{}'.format(n))
    return Nimg


##### Applying Canny Edge Detection ...
def get_canny_edge(img,low,high,n):
    can = np.zeros(img.shape)
    for i in range(img.shape[0]):
        new = np.uint8(img[i])
        new = cv2.Canny(new,low,high,L2gradient=True)
        can[i] = new
        print('Slice {}-{}'.format(n,i))
    return can
