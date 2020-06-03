from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

from matplotlib import style
from matplotlib.widgets import Cursor, Button
import pandas as pd
import mplcursors
#%matplotlib inline
style.use('ggplot')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

xl = pd.ExcelFile("E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\all_GLCM_results.xlsx")


dataX = [1,2,3,4,5,6,7]
sheets = ['dtree_unique_max','gaussnb_unique_max','kn_unique_max','logreg_unique_max','lda_unique_max','randforest_unique_max','svc_unique_max']

classifiers = []

for i in range(7):
    classifiers.append(pd.read_excel(xl,sheets[i]))

accuracies = []

for i in range(len(classifiers)):
    accuracies.append(list(classifiers[i]['% ACCURACY']*100))

n = int(input('Enter component no. >> '))

xd = list(range(1,8))
yd = []
for i in range(7):
    yd.append(accuracies[i][n-1])

fig, ax = plt.subplots(2)
ax[0].set_title('classifier performance on GLCM data')
ax[0].set_xlabel('Components')

ax[0].set_ylabel('Accuracy %')
ax[1].set_xlabel('Components')
ax[1].set_ylabel('Accuracy %')

ax[0].scatter(xd, yd, alpha=0.5, s=7)
ax[1].plot(xd, yd, alpha=0.5)

mplcursors.cursor(hover=True)
plt.show()