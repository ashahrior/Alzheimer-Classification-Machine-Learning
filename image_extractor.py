import numpy as np
from skimage import io
from pathlib import Path
from functional_modules import data_viewer_module as dv
import matplotlib.pyplot as plt
import cv2
from functional_modules import feature_computation_module as fc
import time

ad_npy = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\AD_mainNPY"
cn_npy = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\CN_mainNPY"
mci_npy = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\MCI_mainNPY"

images = r"E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\IMAGES\\"


def modify_image_orientation(case, files):
    for x in files:
        fp = f"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\{case}_mainNPY\data{x}.npy"

        data = np.load(fp, allow_pickle=True)

        data = np.flip(data, 0)
        data = np.rot90(data, k=3, axes=(0, 1))

        t = images+"\\data{}.npy"
        np.save(t.format(x), data)
        d = np.load(t.format(x), allow_pickle=True)
        dv.Show(d)
        


def slicer(src, case, limit=54):
    for i in range(1, limit+1):
        print(f'{case} NPY Data {i}')
        file_path = src + '\data{}.npy'
        data = np.load(file_path.format(i), allow_pickle=True)
        x = 1
        l, h = fc.get_high_low_gray_level(data)
        data = fc.change_image_dynamic_range(data, i, l, h)
        for d in data:
            imgloc = images + case + '\\{}-Data{}\\'.format(case, i)
            Path(imgloc).mkdir(parents=True, exist_ok=True)
            imgfile = imgloc + f'{case}_{i}_img{x}.jpg'
            cv2.imwrite(imgfile, d)
            x += 1
    return

def slicer2(src, case, file_number):
    for i in file_number:
        print(f'{case} NPY Data {i}')
        file_path = src + '\data{}.npy'
        data = np.load(file_path.format(i), allow_pickle=True)
        x = 1
        l, h = fc.get_high_low_gray_level(data)
        data = fc.change_image_dynamic_range(data, i, l, h)
        for d in data:
            imgloc = images + case + '\\{}-Data{}\\'.format(case, i)
            Path(imgloc).mkdir(parents=True, exist_ok=True)
            imgfile = imgloc + f'{case}_{i}_img{x}.jpg'
            cv2.imwrite(imgfile, d)
            x += 1
    return


start_time = time.time()
'''
slicer(ad_npy, 'AD', 54)
slicer(cn_npy, 'CN', 115)
print('CN complete')
slicer(mci_npy, 'MCI', 133)
print('MCI complete')
'''
slicer2(mci_npy,'MCI',[50,59])
#modify_image_orientation('MCI', [50,59])
e = int(time.time() - start_time)
print('Time elapsed- {:02d}:{:02d}:{:02d}'.format(e //3600, (e % 3600 // 60), e % 60))
