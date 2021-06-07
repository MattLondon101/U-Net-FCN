import tensorflow.keras.backend as keras
import skimage.io as io
import os
import skimage.transform as trans
import numpy as np
from natsort import natsorted,ns
from sklearn.metrics import jaccard_score
import cv2 as cv

# Jaccard index for training
def jaccard_index(y_true, y_pred):
    intersection = keras.sum(keras.abs(y_true * y_pred), axis=-1)
    sum_ = keras.sum(keras.abs(y_true) + keras.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return jac


# Jaccard index for testing
def jaccard_test(predir,labdir):
    predl=[os.path.join(predir,i) for i in natsorted(os.listdir(predir),alg=ns.PATH)]
    labl=[os.path.join(labdir,i) for i in natsorted(os.listdir(labdir),alg=ns.PATH)]
    jcs=[]
    for i,j in zip(predl,labl):
        im1=cv.imread(i,0)
        im2=cv.imread(j,0)
        im1 = np.asarray(im1).astype(bool)
        im2 = np.asarray(im2).astype(bool)
        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        im1=im1.flatten()
        im2=im2.flatten()
        jc=jaccard_score(im1,im2)
        jcs.append(jc)
    return jcs

  
