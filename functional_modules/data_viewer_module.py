import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import cv2


class IndexTracker(object):
    def __init__(self, ax, X, view):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, self.rows, self.cols = X.shape
        self.view = view
        self.rem = 1

        if self.view == 1:
            self.ind = self.slices//2
            self.rem = self.slices
            self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray')
            self.update()
        elif self.view == 2:
            self.ind = self.rows//2
            self.rem = self.rows
            self.im = ax.imshow(self.X[:, self.ind, :], cmap='gray')
            self.update()
        else:
            self.ind = self.cols//2
            self.rem = self.cols
            self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
            self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.rem
        else:
            self.ind = (self.ind - 1) % self.rem
        self.update()

    def update(self):
        if self.view == 1:
            self.im.set_data(self.X[self.ind, :, :])
        elif self.view == 2:
            self.im.set_data(self.X[:, self.ind, :])
        else:
            self.im.set_data(self.X[:, :, self.ind])

        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def Show(img, view):

    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, img, view)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def ShowHist(img):
    plt.hist(img.ravel(), 256)
    plt.show()


# img = nib.load(
#     f"C:\\Users\\ashah\\Desktop\\THESIS_WRITING\\New folder\\x (2).nii")
# print(img.shape)

# print(img)

# img = img.get_data()
# print(img.shape)

# Show(img, 3)
