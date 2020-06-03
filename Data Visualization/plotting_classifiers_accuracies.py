# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

from matplotlib import style
from matplotlib.widgets import Cursor, Button
import pandas as pd
import mplcursors
style.use('ggplot')
#%matplotlib inline

# %%
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))

# %%
def single_plotter(xl, sheet, typee):
    
    data = pd.read_excel(xl,sheet)
    fig, ax = plt.subplots(2)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    ax[0].scatter(data['COMPONENT-NO.'], data['% ACCURACY'],alpha=0.5, s=7)
    ax[0].set_xlabel('Components')
    ax[0].set_ylabel('Accuracy %')
    ax[0].set_title(typee + 'classifier performance on GLCM data')
    ax[1].plot(data['COMPONENT-NO.'], data['% ACCURACY'], alpha = 0.5)
    ax[1].set_xlabel('Components')
    ax[1].set_ylabel('Accuracy %')
    mplcursors.cursor(hover=True)
    plt.show()
    #plt.savefig(path+'{}.png'.format(serial))

# %%
def all_plotter(dt,gb,k,lg,ld,r,s):
    xl = pd.ExcelFile("E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\all_GLCM_results.xlsx")
    sheets = ['dtree_unique_max','gaussnb_unique_max','kn_unique_max','logreg_unique_max','lda_unique_max','randforest_unique_max','svc_unique_max']
    
    dtree = pd.read_excel(xl,sheets[0])
    gnb = pd.read_excel(xl,sheets[1])
    kn = pd.read_excel(xl,sheets[2])
    log = pd.read_excel(xl,sheets[3])
    lda = pd.read_excel(xl,sheets[4])
    rf = pd.read_excel(xl,sheets[5])
    sv = pd.read_excel(xl,sheets[6])
    
    fig, ax = plt.subplots(2)
    ax[0].set_title('classifier performance on GLCM data')
    
    ax[0].set_xlabel('Components')
    ax[0].set_ylabel('Accuracy %')
    ax[1].set_xlabel('Components')
    ax[1].set_ylabel('Accuracy %')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    if dt:
        ax[0].scatter(dtree['COMPONENT-NO.'], dtree['% ACCURACY'], alpha=0.5, s=7)
        ax[1].plot(dtree['COMPONENT-NO.'], dtree['% ACCURACY'], alpha=0.5)
    if gb:
        ax[0].scatter(gnb['COMPONENT-NO.'], gnb['% ACCURACY'], alpha=0.5, s=7)
        ax[1].plot(gnb['COMPONENT-NO.'], gnb['% ACCURACY'], alpha=0.5)
    if k:
        ax[0].scatter(kn['COMPONENT-NO.'], kn['% ACCURACY'], alpha=0.5, s=7)
        ax[1].plot(kn['COMPONENT-NO.'], kn['% ACCURACY'], alpha=0.5)
    if lg:
        ax[0].scatter(log['COMPONENT-NO.'], log['% ACCURACY'], alpha=0.5, s=7)
        ax[1].plot(log['COMPONENT-NO.'], log['% ACCURACY'], alpha=0.5)
    if ld:
        ax[0].scatter(lda['COMPONENT-NO.'], lda['% ACCURACY'], alpha=0.5, s=7)
        ax[1].plot(lda['COMPONENT-NO.'], lda['% ACCURACY'], alpha=0.5)
    if r:
        ax[0].scatter(rf['COMPONENT-NO.'], rf['% ACCURACY'],alpha=0.5, s=7)
        ax[1].plot(rf['COMPONENT-NO.'], rf['% ACCURACY'],alpha=0.5)
    if s:
        ax[0].scatter(sv['COMPONENT-NO.'], sv['% ACCURACY'], alpha=0.5, s=7)
        ax[1].plot(sv['COMPONENT-NO.'], sv['% ACCURACY'], alpha=0.5)
    
    mplcursors.cursor(hover=True)
    plt.show()
    #plt.savefig(path+'{}.png'.format(serial))

# %%
if __name__ == "__main__":
    df = pd.ExcelFile("E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\all_GLCM_results.xlsx")
    classifiers = ['dtree_unique_max','gaussnb_unique_max','kn_unique_max','logreg_unique_max','lda_unique_max','randforest_unique_max','svc_unique_max']
    #single_plotter(df,'dtree_unique_max','Decision tree')
    #single_plotter(df,'gaussnb_unique_max','Gaussian NB')
    #single_plotter(df,'kn_unique_max','KNeighbor')
    #single_plotter(df,'lda_unique_max','LDA')
    #single_plotter(df,'logreg_unique_max','Logistic regression')
    single_plotter(df,'randforest_unique_max','Random forest')
    #single_plotter(df,'svc_unique_max','SVC')
    
    #interact(all_plotter,dt = True, gb = False, k = False, lg = False, ld = False, r= False, s= False)



