import os
from pathlib import Path

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

#ad-3-113

refp = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\IMAGES\AD\AD-Data3\AD_3_img113.jpg"

src_dir = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\IMAGES\{}\\"

targ_dir = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\IMAGES\Histogram-matched\{}\\"

template = mpimg.imread(refp)

case = 'AD'

def display_results(template, image, matched):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(image, cmap=plt.get_cmap('gray'))
    ax1.set_title('Source')
    ax2.imshow(template, cmap=plt.get_cmap('gray'))
    ax2.set_title('Template')
    ax3.imshow(matched, cmap=plt.get_cmap('gray'))
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show()
    return

for i in range(1, 55):
    curr_dir = src_dir.format(case) + '{}-Data{}\\'.format(case, i)
    os.chdir(curr_dir)
    print('In directory- ',os.getcwd())
    count = 1
    for file in os.listdir():
        print(file, '->')
        file_name, file_ext = os.path.splitext(file)
        img = mpimg.imread(file)
        print('Opened - ',end=' ')
        matched = match_histograms(img, template, multichannel=False)
        print('Matching done')
        #display_results(template, img, matched)
        imgloc = targ_dir.format(case) + '{}-Data{}\\'.format(case, i)
        Path(imgloc).mkdir(parents=True, exist_ok=True)
        write_img = imgloc+ file_name + file_ext
        cv2.imwrite(write_img, matched)
        print('Writing done.')
        count += 1
    print(count-1,'files written.')

#plt.imshow(template, cmap=plt.get_cmap('gray'))
#plt.show()
