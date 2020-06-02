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
#%matplotlib notebook

xl = pd.ExcelFile("E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\all_GLCM_results.xlsx")


dataX = [1,2,3,4,5,6,7]
sheets = ['dtree_unique_max','gaussnb_unique_max','kn_unique_max','logreg_unique_max','lda_unique_max','randforest_unique_max','svc_unique_max']

dtree = pd.read_excel(xl,sheets[0])
gnb = pd.read_excel(xl,sheets[1])
kn = pd.read_excel(xl,sheets[2])
log = pd.read_excel(xl,sheets[3])
lda = pd.read_excel(xl,sheets[4])
rf = pd.read_excel(xl,sheets[5])
sv = pd.read_excel(xl,sheets[6])


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

dataY = np.random.rand(100)
dataZ = np.random.rand(100)

#ax.plot(dataX, dataY, dataZ, c='r', marker='^', alpha=0.5)

ax.scatter(dataX[0], dtree['COMPONENT-NO.'], dtree['% ACCURACY'], alpha=0.5)

ax.scatter(dataX[1], gnb['COMPONENT-NO.'], gnb['% ACCURACY'], alpha=0.5)

ax.scatter(dataX[2], kn['COMPONENT-NO.'], kn['% ACCURACY'], alpha=0.5)

ax.scatter(dataX[3], log['COMPONENT-NO.'], log['% ACCURACY'], alpha=0.5)

ax.scatter(dataX[4], lda['COMPONENT-NO.'], lda['% ACCURACY'], alpha=0.5)

ax.scatter(dataX[5], rf['COMPONENT-NO.'], rf['% ACCURACY'],alpha=0.5)

ax.scatter(dataX[6], sv['COMPONENT-NO.'], sv['% ACCURACY'], alpha=0.5)

ax.set_xticks(dataX, minor=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
