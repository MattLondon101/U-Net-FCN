from __future__ import print_function
import subprocess
subprocess.check_call(['python','-m','pip','install','tensorflow==1.15'])
subprocess.check_call(['python','-m','pip','install','opencv-python'])
subprocess.check_call(['python','-m','pip','install','keras==2.3'])
subprocess.check_call(['python','-m','pip','install','matplotlib'])
subprocess.check_call(['python','-m','pip','install','natsort'])
subprocess.check_call(['python','-m','pip','install','scikit-image'])
subprocess.check_call(['python','-m','pip','install','pandas'])
subprocess.check_call(['python','-m','pip','install','--user','h5py==2.10.0','--force-reinstall','--no-cache-dir'])
subprocess.check_call(['python','-m','pip','install','scikit-learn'])
subprocess.check_call(['python','-m','pip','install','Pillow'])
subprocess.check_call(['python','-m','pip','install','natsort'])
subprocess.check_call(['python','-m','pip','install','--user','numpy==1.19.5','--force-reinstall','--no-cache-dir'])
subprocess.check_call(['python','-m','pip','install','natsort'])
subprocess.check_call(['python','-m','pip','install','jupyter'])
subprocess.check_call(['python','-m','pip','install','--upgrade','tensorflow-model-optimization'])


import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted,ns
import numpy as np
import pandas as pd
import io
import h5py
import skimage.transform as trans
import skimage.io as io
import glob
from keras.preprocessing.image import ImageDataGenerator
import shutil
from sys import getsizeof
import zipfile

gs=getsizeof
mean=np.mean
nu=np.unique

def imsh(img,nrows=1,ncols=1,cmap='gray'):
    fig,ax=plt.subplots(nrows=nrows,ncols=ncols,
    figsize=(5,5))
    ax.imshow(img,cmap='gray')
    ax.axis('off')
    return (plt.show())

# unzip files
with zipfile.ZipFile('datadir.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

with zipfile.ZipFile('logs.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

