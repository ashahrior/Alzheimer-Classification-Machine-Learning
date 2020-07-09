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

################ Applying K-means Algorithm for K = 3 ......
def run_KMeans_algo(asm,cenasm,brightness,cenbrtns,dissimilarity,cendiss\
                  ,entropy,cenentr,homo,cenhomo,idm,cenidm,variance,cenvar,n):
    c1,c2,c3 = 5,10,15
    ASM = load_features(asm,n)
    ASM = normalize_features(ASM)
    print('DoneASM')
    c1ASM = ASM[c1] ### Setting the seeds manually and arbitraryly ..
    c2ASM = ASM[c2]
    c3ASM = ASM[c3]

    Brtns = load_features(brightness,n)
    Brtns = normalize_features(Brtns)
    print('DoneBrtns')
    c1Brtns = Brtns[c1]
    c2Brtns = Brtns[c2]
    c3Brtns = Brtns[c3]

    Diss = load_features(dissimilarity,n)
    Diss = normalize_features(Diss)
    print('DoneDiss')
    c1Diss = Diss[c1]
    c2Diss = Diss[c2]
    c3Diss = Diss[c3]

    Entr = load_features(entropy,n)
    Entr = normalize_features(Entr)
    print('DoneEntr')
    c1Entr = Entr[c1]
    c2Entr = Entr[c2]
    c3Entr = Entr[c3]

    Homo = load_features(homo,n)
    Homo = normalize_features(Homo)
    print('DoneHomo')
    c1Homo = Homo[c1]
    c2Homo = Homo[c2]
    c3Homo = Homo[c3]

    Idm = load_features(idm,n)
    Idm = normalize_features(Idm)
    print('DoneIdm')
    c1Idm = Idm[c1]
    c2Idm = Idm[c2]
    c3Idm = Idm[c3]

    Var = load_features(variance,n)
    Var = normalize_features(Var)
    print('DoneVar')
    c1Var = Var[c1]
    c2Var = Var[c2]
    c3Var = Var[c3]

    #print(ASM[0]*1000000)
    #gjkhg = input('Enter : ')


    Range = 10
    for case in range(Range):
        cluster1 = []
        cluster2 = []
        cluster3 = []

        for i in range(n):
            #asm,brightness,dissimilarity,entropy,homo,idm,variance
            asmd1,asmd2,asmd3 = get_centroidal_distance(c1ASM,c2ASM,c3ASM,ASM[i])
            brtnsd1,brtnsd2,brtnsd3 = get_centroidal_distance(c1Brtns,c2Brtns,c3Brtns,Brtns[i])
            dissd1,dissd2,dissd3 = get_centroidal_distance(c1Diss,c2Diss,c3Diss,Diss[i])
            entrd1,entrd2,entrd3 = get_centroidal_distance(c1Entr,c2Entr,c3Entr,Entr[i])
            homod1,homod2,homod3 = get_centroidal_distance(c1Homo,c2Homo,c3Homo,Homo[i])
            idmd1,idmd2,idmd3 = get_centroidal_distance(c1Idm,c2Idm,c3Idm,Idm[i])
            vard1,vard2,vard3 = get_centroidal_distance(c1Var,c2Var,c3Var,Var[i])

            totalDist1 = sum(asmd1)+sum(brtnsd1)+sum(dissd1)+sum(entrd1)+sum(homod1)\
                        +sum(idmd1)+sum(vard1)
            totalDist2 = sum(asmd2)+sum(brtnsd2)+sum(dissd2)+sum(entrd2)+sum(homod2)\
                        +sum(idmd2)+sum(vard2)
            totalDist3 = sum(asmd3)+sum(brtnsd3)+sum(dissd3)+sum(entrd3)+sum(homod3)\
                        +sum(idmd3)+sum(vard3)
            print('Data{}'.format(i),totalDist1,totalDist2,totalDist3)

            '''
            if i%5 == 0:
                wait = input('Wait for me : ')
            '''
            cluster_no = 0

            if totalDist1 >= totalDist2:
                if totalDist2 >= totalDist3:
                    cluster_no = 3
                    cluster3.append(i)
                else:
                    cluster_no = 2
                    cluster2.append(i)
            else:
                if totalDist1 >= totalDist3:
                    cluster_no = 3
                    cluster3.append(i)
                else:
                    cluster_no = 1
                    cluster1.append(i)

        print(cluster1)
        print(cluster2)
        print(cluster3)

        wait = input('Wait for Me : ')

        ####### Computing new centroid for cluster 1.
        oldc1ASM = c1ASM
        c1ASM = ASM[cluster1[0]]
        oldc1Brtns = c1Brtns
        c1Brtns = Brtns[cluster1[0]]
        oldc1Diss = c1Diss
        c1Diss = Diss[cluster1[0]]
        oldc1Entr = c1Entr
        c1Entr = Entr[cluster1[0]]
        oldc1Homo = c1Homo
        c1Homo = Homo[cluster1[0]]
        oldc1Idm = c1Idm
        c1Idm = Idm[cluster1[0]]
        oldc1Var = c1Var
        c1Var = Var[cluster1[0]]

        for i in range(1,len(cluster1)):
            #asm,brightness,dissimilarity,entropy,homo,idm,variance
            c1ASM = (c1ASM + ASM[cluster1[i]])/2
            c1Brtns = (c1Brtns + Brtns[cluster1[i]])/2
            c1Diss = (c1Diss + Diss[cluster1[i]])/2
            c1Entr = (c1Entr + Entr[cluster1[i]])/2
            c1Homo = (c1Homo + Homo[cluster1[i]])/2
            c1Idm = (c1Idm + Idm[cluster1[i]])/2
            c1Var = (c1Var + Var[cluster1[i]])/2

        ####### Computing new centroid for cluster 2.
        oldc2ASM = c2ASM
        c2ASM = ASM[cluster2[0]]
        oldc2Brtns = c2Brtns
        c2Brtns = Brtns[cluster2[0]]
        oldc2Diss = c2Diss
        c2Diss = Diss[cluster2[0]]
        oldc2Entr = c2Entr
        c2Entr = Entr[cluster2[0]]
        oldc2Homo = c2Homo
        c2Homo = Homo[cluster2[0]]
        oldc2Idm = c2Idm
        c2Idm = Idm[cluster2[0]]
        oldc2Var = c2Var
        c2Var = Var[cluster2[0]]

        for i in range(1,len(cluster2)):
            #asm,brightness,dissimilarity,entropy,homo,idm,variance
            c2ASM = (c2ASM + ASM[cluster2[i]])/2
            c2Brtns = (c2Brtns + Brtns[cluster2[i]])/2
            c2Diss = (c2Diss + Diss[cluster2[i]])/2
            c2Entr = (c2Entr + Entr[cluster2[i]])/2
            c2Homo = (c2Homo + Homo[cluster2[i]])/2
            c2Idm = (c2Idm + Idm[cluster2[i]])/2
            c2Var = (c2Var + Var[cluster2[i]])/2

        ####### Computing new centroid for cluster 3.
        oldc3ASM = c3ASM
        c3ASM = ASM[cluster3[0]]
        oldc3Brtns = c3Brtns
        c3Brtns = Brtns[cluster3[0]]
        oldc3Diss = c3Diss
        c3Diss = Diss[cluster3[0]]
        oldc3Entr = c3Entr
        c3Entr = Entr[cluster3[0]]
        oldc3Homo = c3Homo
        c3Homo = Homo[cluster3[0]]
        oldc3Idm = c3Idm
        c3Idm = Idm[cluster3[0]]
        oldc3Var = c3Var
        c3Var = Var[cluster3[0]]

        for i in range(1,len(cluster3)):
            #asm,brightness,dissimilarity,entropy,homo,idm,variance
            c3ASM = (c3ASM + ASM[cluster3[i]])/2
            c3Brtns = (c3Brtns + Brtns[cluster3[i]])/2
            c3Diss = (c3Diss + Diss[cluster3[i]])/2
            c3Entr = (c3Entr + Entr[cluster3[i]])/2
            c3Homo = (c3Homo + Homo[cluster3[i]])/2
            c3Idm = (c3Idm + Idm[cluster3[i]])/2
            c3Var = (c3Var + Var[cluster3[i]])/2


        #asm,brightness,dissimilarity,entropy,homo,idm,variance
        sumC1 = abs(c1ASM-oldc1ASM)+abs(c1Brtns-oldc1Brtns)+abs(c1Diss-oldc1Diss)\
                +abs(c1Entr-oldc1Entr)+abs(c1Homo-oldc1Homo)+abs(c1Idm-oldc1Idm)\
                +abs(c1Var-oldc1Var)

        sumC2 = abs(c2ASM-oldc2ASM)+abs(c2Brtns-oldc2Brtns)+abs(c2Diss-oldc2Diss)\
                +abs(c2Entr-oldc2Entr)+abs(c2Homo-oldc2Homo)+abs(c2Idm-oldc2Idm)\
                +abs(c2Var-oldc2Var)

        sumC3 = abs(c3ASM-oldc3ASM)+abs(c3Brtns-oldc3Brtns)+abs(c3Diss-oldc3Diss)\
                +abs(c3Entr-oldc3Entr)+abs(c3Homo-oldc3Homo)+abs(c3Idm-oldc3Idm)\
                +abs(c3Var-oldc3Var)

        print(sum(sumC1),sum(sumC2),sum(sumC3))
        wait = input('Wait for Me : ')

        del cluster1
        del cluster2
        del cluster3

        #asm,brightness,dissimilarity,entropy,homo,idm,variance
        if sum(sumC1) <= 1 and sum(sumC2) <= 1 and sum(sumC3) <= 1 :

            np.save(cenasm.format(1),c1ASM)
            np.save(cenasm.format(2),c2ASM)
            np.save(cenasm.format(3),c3ASM)

            np.save(cenbrtns.format(1),c1Brtns)
            np.save(cenbrtns.format(2),c2Brtns)
            np.save(cenbrtns.format(3),c3Brtns)

            np.save(cendiss.format(1),c1Diss)
            np.save(cendiss.format(2),c2Diss)
            np.save(cendiss.format(3),c3Diss)

            #asm,brightness,dissimilarity,entropy,homo,idm,variance
            np.save(cenentr.format(1),c1Entr)
            np.save(cenentr.format(2),c2Entr)
            np.save(cenentr.format(3),c3Entr)

            np.save(cenhomo.format(1),c1Homo)
            np.save(cenhomo.format(2),c2Homo)
            np.save(cenhomo.format(3),c3Homo)

            np.save(cenidm.format(1),c1Idm)
            np.save(cenidm.format(2),c2Idm)
            np.save(cenidm.format(3),c3Idm)

            np.save(cenvar.format(1),c1Var)
            np.save(cenvar.format(2),c2Var)
            np.save(cenvar.format(3),c3Var)

            print('Centroids Saved')
            break

'''
####### predicting a data set ..
# n is the serial number of dataset
def getGroup(asmloc,homoloc,dissloc,idmloc,entrloc,brtnsloc,varloc,n):

    asm = normFeature(OpenNpy(asmloc,n))
    homo = normFeature(OpenNpy(homoloc,n))
    diss = normFeature(OpenNpy(dissloc,n))
    idm = normFeature(OpenNpy(idmloc,n))
    entr = normFeature(OpenNpy(entrloc,n))
    brtns = normFeature(OpenNpy(brtnsloc,n))
    var = normFeature(OpenNpy(varloc,n))

    print('Features Loaded !!')
    ### Opening The centroids and checking the distance ...

    ### Testing for MCI ..
    distanceMCI = []
    for i in range(3):
        asmc = OpenNpy(loc.mciAsmcen,i+1)
        homoc = OpenNpy(loc.mciHomocen,i+1)
        dissc = OpenNpy(loc.mciDisscen,i+1)
        idmc = OpenNpy(loc.mciIdmcen,i+1)
        entrc = OpenNpy(loc.mciEntrcen,i+1)
        brtnsc = OpenNpy(loc.mciBrtnscen,i+1)
        varc = OpenNpy(loc.mciVarcen,i+1)

        asmd = abs(asmc - asm)
        homod = abs(homoc - homo)
        dissd = abs(dissc - diss)
        idmd = abs(idmc - idm)
        entrd = abs(entrc - entr)
        brtnsd = abs(brtnsc - brtns)
        vard = abs(varc - var)

        totalDistance = sum(asmd)+sum(homod)+sum(dissd)+sum(idmd)+sum(entrd)+sum(brtnsd)+sum(vard)
        distanceMCI.append(totalDistance)

    distanceMCI.sort()
    print(distanceMCI)
    print('Done for MCI !!')

    ### Testing for AD ..
    distanceAD = []
    for i in range(3):
        asmc = OpenNpy(loc.adAsmcen,i+1)
        homoc = OpenNpy(loc.adHomocen,i+1)
        dissc = OpenNpy(loc.adDisscen,i+1)
        idmc = OpenNpy(loc.adIdmcen,i+1)
        entrc = OpenNpy(loc.adEntrcen,i+1)
        brtnsc = OpenNpy(loc.adBrtnscen,i+1)
        varc = OpenNpy(loc.adVarcen,i+1)

        asmd = abs(asmc - asm)
        homod = abs(homoc - homo)
        dissd = abs(dissc - diss)
        idmd = abs(idmc - idm)
        entrd = abs(entrc - entr)
        brtnsd = abs(brtnsc - brtns)
        vard = abs(varc - var)

        totalDistance = sum(asmd)+sum(homod)+sum(dissd)+sum(idmd)+sum(entrd)+sum(brtnsd)+sum(vard)
        distanceAD.append(totalDistance)

    distanceAD.sort()
    print(distanceAD)
    print('Done for AD !!')

    ### Testing for CN ..
    distanceCN = []
    for i in range(3):
        asmc = OpenNpy(loc.cnAsmcen,i+1)
        homoc = OpenNpy(loc.cnHomocen,i+1)
        dissc = OpenNpy(loc.cnDisscen,i+1)
        idmc = OpenNpy(loc.cnIdmcen,i+1)
        entrc = OpenNpy(loc.cnEntrcen,i+1)
        brtnsc = OpenNpy(loc.cnBrtnscen,i+1)
        varc = OpenNpy(loc.cnVarcen,i+1)

        asmd = abs(asmc - asm)
        homod = abs(homoc - homo)
        dissd = abs(dissc - diss)
        idmd = abs(idmc - idm)
        entrd = abs(entrc - entr)
        brtnsd = abs(brtnsc - brtns)
        vard = abs(varc - var)

        totalDistance = sum(asmd)+sum(homod)+sum(dissd)+sum(idmd)+sum(entrd)+sum(brtnsd)+sum(vard)
        distanceCN.append(totalDistance)

    distanceCN.sort()
    print(distanceCN)
    print('Done for CN !!')

    ### Computing Percentage
    x = distanceAD[0]
    y = distanceCN[0]
    z = distanceMCI[0]

    total = x+y+z
    x = total/x
    y = total/y
    z = total/z

    total = x+y+z
    x = (x*100)/total
    y = (y*100)/total
    z = (z*100)/total

    if distanceAD[0] <= distanceCN[0] :
        if distanceAD[0] <= distanceMCI[0]:
            return 1,x,y,z                    ### Means AD
        else:
            return 3,x,y,z                    ### Means MCI
    else:
        if distanceCN[0] <= distanceMCI[0]:
            return 2,x,y,z                    ### Means CN
        else:
            return 3,x,y,z                    ### Means MCI
'''


######################################################################################
######################################################################################
############### Computing Entropy ....
def get_entropy_GLCM(img,n):
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
