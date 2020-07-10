import time
from pathlib import Path
import os
import re

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io

from functional_modules import data_viewer_module as dv
from functional_modules import feature_computation_module as fc

source_folder = ''
target_folder = ''

images_folder = r"E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\IMAGES\\"

npy_norm_targ = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\Normalized_NPY\{}_normNPY\\"

selected_files = {
    'AD': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'CN': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 22, 24, 25, 28, 29, 30, 33, 39, 40, 41, 49, 59, 60, 63, 64, 70, 71, 72, 73, 74, 78, 79, 80, 81, 82, 84, 85, 87, 99, 100, 101, 106, 109, 110, 111, 112, 115],

    'MCI': [1,6,7,8,9,10,11,27,29,30,31,32,33,34,36,40,43,44,45,46,52,55,56,57,58,59,60,61,62,63,65,66,67,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,96,98,99,113,114,]
}


ad_data_tuples = {
    1: (54, 112), 2: (68, 121), 3: (74, 125), 4: (74, 124), 5: (64, 118), 6: (82, 132), 7: (74, 124), 8: (74, 128), 9: (72, 122), 10: (72, 125), 11: (56, 110), 12: (59, 113), 13: (68, 124), 14: (70, 124), 15: (65, 118), 16: (68, 118), 17: (91, 141), 18: (62, 108), 19: (68, 122), 20: (72, 126), 21: (46, 108), 22: (72, 128), 23: (82, 124), 24: (54, 110), 25: (84, 142), 26: (70, 118), 27: (60, 114), 28: (70, 114), 29: (92, 145), 30: (64, 116), 31: (56, 106), 32: (68, 120), 33: (71, 117), 34: (78, 124), 35: (64, 117), 36: (84, 140), 37: (74, 134), 38: (54, 106), 39: (54, 106), 40: (74, 134), 41: (82, 138), 42: (61, 128), 43: (68, 128), 44: (46, 110), 45: (82, 127), 46: (62, 130), 47: (100, 146), 48: (54, 114), 49: (52, 96), 50: (80, 126), 51: (84, 134), 52: (68, 114), 53: (66, 114), 54: (80, 140)
}

mci_data_tuples = {
    1: (63, 112), 2: (78, 127), 3: (84, 133), 4: (42, 82), 5: (77, 127), 6: (67, 126), 7: (61, 118), 8: (64, 124), 9: (57, 114), 10: (70, 130), 11: (80, 138), 12: (85, 124), 13: (81, 133), 14: (59, 97), 15: (79, 112), 16: (66, 109), 17: (67, 103), 18: (47, 89), 19: (49, 95), 20: (58, 92), 21: (55, 93), 22: (67, 97), 23: (73, 103), 24: (84, 128), 25: (61, 91), 26: (72, 98), 27: (58, 114), 28: (48, 90), 29: (67, 121), 30: (55, 106), 31: (81, 131), 32: (60, 116), 33: (79, 131), 34: (61, 114), 35: (79, 105), 36: (55, 105), 37: (61, 85), 38: (55, 100), 39: (61, 82), 40: (97, 142), 41: (62, 95), 42: (79, 115), 43: (61, 115), 44: (64, 111), 45: (98, 146), 46: (67, 110), 47: (93, 123), 48: (85, 114), 49: (84, 126), 50: (71, 126), 51: (55, 110), 52: (90, 140), 53: (56, 108), 54: (56, 106), 55: (48, 112), 56: (47, 112), 57: (88, 133), 58: (55, 119), 59: (68, 134), 60: (42, 106), 61: (46, 106), 62: (79, 136), 63: (86, 130), 64: (78, 115), 65: (82, 126), 66: (81, 125), 67: (51, 96), 68: (52, 85), 69: (85, 119), 70: (79, 108), 71: (80, 100), 72: (73, 94), 73: (85, 99), 74: (74, 129), 75: (75, 124), 76: (71, 118), 77: (59, 113), 78: (75, 125), 79: (65, 110), 80: (71, 125), 81: (77, 125), 82: (71, 123), 83: (60, 120), 84: (56, 113), 85: (76, 133), 86: (81, 135), 87: (90, 126), 88: (72, 133), 89: (82, 141), 90: (55, 97), 91: (89, 130), 92: (96, 130), 93: (77, 127), 94: (62, 100), 95: (56, 102), 96: (56, 109), 97: (57, 102), 98: (56, 110), 99: (50, 101), 100: (70, 104), 101: (56, 97), 102: (71, 118), 103: (71, 118), 104: (68, 103), 105: (74, 118), 106: (57, 117), 107: (56, 104), 108: (59, 119), 109: (56, 102), 110: (46, 98), 111: (68, 121), 112: (54, 96), 113: (61, 110), 114: (61, 114), 115: (61, 113), 116: (59, 111), 117: (46, 89), 118: (60, 115), 119: (59, 120), 120: (47, 111), 121: (46, 110), 122: (67, 124), 123: (56, 87), 124: (61, 112), 125: (66, 120), 126: (71, 121), 127: (71, 114), 128: (61, 100), 129: (61, 100), 130: (51, 101), 131: (55, 102), 132: (81, 114), 133: (90, 132)
}

cn_data_tuple = {
    {1: (56, 114), 2: (78, 124), 3: (68, 120), 4: (58, 100), 5: (56, 100), 6: (58, 110), 7: (56, 113), 8: (82, 140), 9: (56, 110), 10: (68, 112), 11: (64, 108), 12: (91, 132), 13: (92, 124), 14: (68, 128), 15: (58, 112), 16: (68, 118), 17: (76, 112), 18: (82, 124), 19: (78, 124), 20: (74, 122), 21: (64, 88), 22: (70, 122), 23: (68, 92), 24: (60, 104), 25: (62, 104), 26: (64, 107), 27: (72, 92), 28: (56, 114), 29: (64, 110), 30: (73, 116), 31: (78, 131), 32: (80, 132), 33: (80, 140), 34: (59, 108), 35: (58, 105), 36: (44, 93), 37: (62, 106), 38: (88, 129), 39: (50, 112), 40: (76, 127), 41: (76, 121), 42: (60, 105), 43: (41, 96), 44: (46, 100), 45: (72, 124), 46: (72, 124), 47: (81, 124), 48: (80, 123), 49: (66, 122), 50: (84, 122), 51: (74, 119), 52: (76, 126), 53: (76, 130), 54: (92, 134), 55: (62, 102), 56: (62, 102), 57: (91, 135), 58: (86, 120), 59: (
        68, 128), 60: (71, 131), 61: (75, 116), 62: (57, 105), 63: (69, 123), 64: (56, 110), 65: (64, 102), 66: (62, 104), 67: (62, 103), 68: (61, 103), 69: (68, 110), 70: (75, 125), 71: (64, 112), 72: (77, 121), 73: (64, 115), 74: (62, 114), 75: (53, 99), 76: (50, 98), 77: (61, 104), 78: (66, 117), 79: (65, 115), 80: (65, 117), 81: (69, 119), 82: (56, 101), 83: (84, 121), 84: (53, 108), 85: (69, 120), 86: (61, 105), 87: (69, 116), 88: (81, 117), 89: (61, 96), 90: (79, 120), 91: (69, 108), 92: (89, 127), 93: (94, 129), 94: (91, 123), 95: (82, 115), 96: (75, 115), 97: (76, 118), 98: (63, 115), 99: (62, 105), 100: (65, 106), 101: (77, 120), 102: (61, 82), 103: (85, 100), 104: (78, 94), 105: (76, 111), 106: (81, 131), 107: (69, 115), 108: (88, 130), 109: (75, 129), 110: (61, 116), 111: (79, 136), 112: (61, 115), 113: (61, 98), 114: (69, 105), 115: (75, 121)}
}



def nii_to_npy_convert(srcfol, targfol):
    """[Converts the nii files to npy files]

    Args:
        srcfol ([str]): [source folder location of the nii files]
        targfol ([str]): [target folder location for the npy files]
    """
    fc.convert_nii_to_npy(srcfol, targfol)
    return
    

def modify_image_orientation(case, srcfol, files:list):
    """Changes the directional orientation the of the data

    Args:
        case ([str]): ['AD'/'CN'/'MCI']
        srcfol ([str]): [source location of the data]
        files (list): [list of files that need modification]
    """
    for x in files:
        fp = srcfol.format(case)+"\Data{x}.npy"
        data = np.load(fp, allow_pickle=True)
        data = np.flip(data, 0)
        data = np.rot90(data, k=3, axes=(0, 1))
        t = srcfol.format(case)+f"\\Data{x}.npy"
        np.save(t, data)
        #d = np.load(t, allow_pickle=True)
        #dv.Show(d)
    return


def npy_to_img(srcfol: str, case: str):
    """[Convert 3d npy data files to 2d images]

    Args:
        srcfol (str): [source location of the .npy files]
        case (str): ['AD'/"CN'/'MCI']
    """
    os.chdir(srcfol)
    for file in os.listdir():
        fname, fext = os.path.splitext(file)
        print(f'{case} - {file}')
        x = 1
        data = np.load(file)
        l, h = fc.get_high_low_gray_level(data)
        data = fc.change_image_dynamic_range(data, l, h)
        for d in data:
            imgloc = f'{images_folder}{case}\\{case}-{fname}'
            Path(imgloc).mkdir(parents=True, exist_ok=True)
            imgfile = imgloc + f'{case}-{fname}_img{x}.jpg'
            cv2.imwrite(imgfile, d)
            x += 1
    return


def get_img_names():
    """[extracts names from .jpg image files only]

    Returns:
        [str]: [a string containing all the image names]
    """
    titles = ''
    for file in os.listdir():
        if file.__contains__('.jpg'):
            titles += file+'\n'
    return titles


def get_tuples(imgs_folder, case:str, n:int):
    """[extracts the starting and finishing file serial values]

    Args:
        imgs_folder ([string]): [location of the images]
        case ([string]): ['AD'/'CN'/'MCI']
        n ([int]): [number of folders in the location]

    Returns:
        [dictionary]: [returns a dictionary of tuples]
    """
    tuples = {}

    for i in range(1, n+1):
        path = imgs_folder.format(case, case, i)
        os.chdir(path)
        imgs = get_img_names()
        serial = re.findall(r'g[1-9]*[0-9][0-9]', imgs)
        #print(serial)
        serial = [int(i[1:]) for i in serial]
        a, b = min(serial), max(serial)
        #print(i, f'- ({a},{b})')
        tuples[i] = (a, b)
    return tuples


def get_interest_data(src, targ, tuples: dict, case: str):
    """extract the interested part of the whole .npy data

    Args:
        src ([str]): [source location of the main .npy files]
        targ ([str]): [target location of the interested .npy files]
        tuples (dict): [dict holding the interest range tupls]
        case (str): ['AD'/'CN'/'MCI']
    """
    os.chdir(src.format(case))
    for file in os.listdir():
        data = np.load(file, allow_pickle=True)
        index = int(re.findall('\d+', file)[0])
        tup = tuples[index]
        i_data = data[tup[0] : tup[1] + 1, :, :]
        print(f'{case} {file} with interest space {tup} loaded', end=' - ')
        output = targ.format(case) + f'{file}'
        np.save(output, i_data)
        print('saved')
    return


def convert_img_to_npy(case, tuples: dict):
    """convert 2d image files into a 3d npy file

    Args:
        case ([str]): ['AD'/'CN'/'MCI']
        tuples (dict): [a dictionary of tuples for selected range]
    """
    for serial in selected_files[case]:
        folder = f'{images_folder}{case}\\{case}-Data{serial}\\'
        start, finish = tuples[serial]
        images = []
        for slices in range(start, finish + 1):
            file = folder + f'{case}_{serial}_img{slices}.jpg'
            img = mpimg.imread(file)
            print(f'{case}_{serial}_img{slices}.jpg loaded')
            images.append(img)
        data = np.array(img)
        npy_save = npy_norm_targ.format(case) + f'{case}-Data{serial}.npy'
        np.save(npy_save, data)
        print(f'{case}-{serial} npy file saved.\n')
    return


def normalize_data(src, targ, case):
    """[normalize .npy data to the range of 0 to 255]

    Args:
        src ([str]): [source location of the concerned npy files]
        targ ([str]): [target location for normalized npy files]
        case ([str]): ['AD'/'CN'/'MCI']
    """
    os.chdir(src.format(case))
    for file in os.listdir():
        print(f'{file} -',end=' ')
        fname,fext = os.path.splitext(file)
        data = np.load(file, allow_pickle=True)
        print('npy loaded',end=' ')
        low, high = fc.get_high_low_gray_level(data)
        data = fc.change_image_dynamic_range(data, low, high)
        print('normalized',end=' ')
        save_as = targ.format(case) + f"\\norm_{file}"
        np.save(save_as, data)
        print('saved')
    return




if __name__ == "__main__":
    pass
