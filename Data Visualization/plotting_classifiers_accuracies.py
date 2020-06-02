# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import matplotlib.pyplot as plt
import matplotlib.artist as artist
from matplotlib import style
from matplotlib.widgets import Cursor, Button
import pandas as pd
import mplcursors
#%matplotlib inline
style.use('ggplot')


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    ('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))

# %%
def plotter(xl, sheet, type):
    
    data = pd.read_excel(xl,sheet)
    fig, ax = plt.subplots(2)

    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    ax[0].scatter(data['COMPONENT-NO.'], data['% ACCURACY'])
    ax[0].set_xlabel('Components')
    ax[0].set_ylabel('Accuracy %')
    ax[0].set_title(type+'classifier performance on GLCM data')
    ax[1].plot(data['COMPONENT-NO.'], data['% ACCURACY'])
    ax[1].set_xlabel('Components')
    ax[1].set_ylabel('Accuracy %')
    mplcursors.cursor(hover=True)
    plt.show()
    #plt.savefig(path+'{}.png'.format(serial))

def multi_plotter(xl,sheets, types):
    data = []
    for i in range(len(sheets)):
        data.append(pd.read_excel(xl,sheets[i]))

# %%
df = pd.ExcelFile("E:\\THESIS\\ADNI_data\\ADNI1_Annual_2_Yr_3T_306_WORK\\LogRegClassifier\\all_GLCM_results.xlsx")

#plotter(df,'dtree_unique_max','Decision tree')
#plotter(df,'gaussnb_unique_max','Gaussian NB')
plotter(df,'kn_unique_max','KNeighbor')
#plotter(df,'lda_unique_max','LDA')
#plotter(df,'logreg_unique_max','Logistic regression')
#plotter(df,'randforest_unique_max','Random forest')
#plotter(df,'svc_unique_max','SVC')




