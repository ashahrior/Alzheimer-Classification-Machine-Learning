from functional_modules import feature_computation_module as fc
import numpy as np

#########################################
'''
: calculate_GLCM_feats(address, location, n, start=0) 
    - Opens up .npy files. 
    - Calculates GLCM features for the .npy files
    - stores the features in separate .npy files
'''
def calculate_GLCM_feats(address, location, n, start=0):
    '''
    :param address: 'folder/folder/' that contains the data*.npy files
    :param location: 'folder/folder/' where feature will be saved
    :param n: number of data file
    :return: nothing but saves the features
    '''
    for i in range(start,n):
        #data = fc.open_interest_data(address+'data{}.npy',i+1) # Openning .npy file from the Location
        data = np.load(address+'data{}.npy'.format(i+1), allow_pickle=True)
        low,high = fc.get_high_low_gray_level(data,i+1) # Retreiving Low and High for the next operation
        data = fc.change_image_dynamic_range(data,i+1,low,high) # getting the changed (Range) Image
        intData = fc.convert_into_integer(data,i+1) # Return Integer Image for GLCM
        glcm = fc.get_GLCM(intData,i+1) # Calculating GLCM
        glcm = fc.normalize_GLCM(glcm,i+1) # Normalizing GLCM

        ### Calculating The features using GLCM ..
        homo = fc.get_homogeneity(glcm,i+1) # Calculating Homo..
        diss = fc.get_dissimilarity(glcm,i+1) # Calculating Dissimilarity
        asm = fc.get_ASM(glcm,i+1) # Calculating ASM
        idm = fc.get_IDM(glcm,i+1) # Calculating IDM

        ### Calculating Features from Actual Image File ..
        entropy = fc.get_entropy(data,i+1) # Calculating Entropy
        brtns = fc.get_brightness(data,i+1) # Calculating Brightness
        variance = fc.get_variance(data,brtns,i+1) # CAlculating Variance

        ### Saving THe Features....
        fc.np.save((location+'homo{}.npy').format(i+1),homo)
        fc.np.save((location+'diss{}.npy').format(i+1),diss)
        fc.np.save((location+'asm{}.npy').format(i+1),asm)
        fc.np.save((location+'idm{}.npy').format(i+1),idm)
        fc.np.save((location+'entropy{}.npy').format(i+1),entropy)
        fc.np.save((location+'brtns{}.npy').format(i+1),brtns)
        fc.np.save((location+'variance{}.npy').format(i+1),variance)



def generate_GLCM_feats_list(address,number_of_files,target,F):
    '''
    :param address: 'folder/folder/' where the features are saved
    :param number_of_files: number of data files
    :param target: 1 for AD, 2 for CN and 3 for MCI
    :param F: Feature Array, just send a list
    :return: returns the updated feature list
    '''
    case_type = ''
    if target==1:
        case_type = 'AD'
    elif target==2:
        case_type = 'CN'
    else: case_type = 'MCI'

    for file_serial in range(1,number_of_files+1):
        #Openning the features for a Data File
        print('Accessing data in {} file #{} >>'.format(case_type,file_serial))
        asm = np.load(address+'asm{}.npy'.format(file_serial),allow_pickle=True)
        #print('ASM for file',file_serial,'done.')
        
        brtns = np.load(address+'brtns{}.npy'.format(file_serial),allow_pickle=True)
        #print('Brightness for file',file_serial,'done.')
        
        diss = np.load(address+'diss{}.npy'.format(file_serial),allow_pickle=True)
        #print('Dissimilarity for file',file_serial,'done.')
        
        entr = np.load(address+'entropy{}.npy'.format(file_serial),allow_pickle=True)
        #print('Entropy for file',file_serial,'done.')
        
        homo = np.load(address+'homo{}.npy'.format(file_serial),allow_pickle=True)
        #print('Homogeneity for file',file_serial,'done.')
        
        idm = np.load(address+'idm{}.npy'.format(file_serial),allow_pickle=True)
        #print('IDM for file',file_serial,'done.')
        
        var = np.load(address+'variance{}.npy'.format(file_serial),allow_pickle=True)
        #print('Variance for file',file_serial,'done.')
        
        print('GLCM retrieved for file #{}  of {} case.'.format(file_serial,case_type))

        #Creating a row for the data file
        row = []
        slices = asm.shape[0]
        for j in range(slices): #slices slices
            row.append(asm[j])
        for j in range(slices):
            row.append(homo[j])
        for j in range(slices):
            row.append(diss[j])
        for j in range(slices):
            row.append(idm[j])
        for j in range(slices):
            row.append(brtns[j])
        for j in range(slices):
            row.append(entr[j])
        for j in range(slices):
            row.append(var[j])
        row.append(target) #Last element is the target
        print('Data row for file #{} created.'.format(file_serial))
        F.append(row)
        print('Data row for file #{} appended to Feature array.'.format(file_serial))
        print()
    return F