import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import cv2


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.rows, self.cols, self.slices = X.shape
        #self.ind = self.slices//2
        #self.ind = self.cols//2
        self.ind = self.rows//2

        self.im = ax.imshow(self.X[self.ind, :,:],cmap='gray')
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) %self.rows
        else:
            self.ind = (self.ind - 1) % self.rows
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind,:,:])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def Show(img):

    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, img)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def ShowHist(img):
    plt.hist(img.ravel(),256)
    plt.show()
