import os
import sys
from natsort import natsorted,ns
import skimage.io as io
import cv2 as cv
import shutil
from PIL import Image
import tensorflow as tf
import time
start_time=time.time()

exec(open('imports.py').read())

from importlib import reload
import data
import model

reload(data)
import data
from data import *
reload(model)
import model
from model import *

# When picking specific non-consecutive images from directory, func:labmv will select labels that correspond with images and copy to new directory.
# written for Carvana dataset (https://www.kaggle.com/c/carvana-image-masking-challenge/data) where jpg images and gif labels have identical names, except labels end with '_mask'
def labmv(srcim,srclab,outlab):
    for fi in natsorted(os.listdir(srcim),alg=ns.PATH):
        fp=os.path.join(srcim,fi)
        if os.path.isfile(fp):
            imname=os.path.splitext(fi)[0]
            labname=imname+'_mask.gif'
            # lbpath=os.path.join(srclab,labname)
            for lfi in natsorted(os.listdir(srclab),alg=ns.PATH):
                if lfi == labname:
                    lbfi=os.path.join(srclab,lfi)
                    shutil.copy(lbfi,os.path.join(outlab,lfi))

labmv('datadir/train/ims','datadir/train/labels_all','datadir/train/labels_gif')


# rename named ims to num
def rimn(src,ext):
    i=0
    for fi in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,fi)
        if os.path.isfile(fp):
            fop=os.path.join(src,str(i)+str(ext))
            os.rename(fp,fop)
            i+=1
rimn('datadir/train/ims','.jpg')


# 3D gif to 2D jpg
def rdgif(src,out,ext):
    i=0
    for fi in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,fi)
        if os.path.isfile(fp):
            im=Image.open(fp)
            pal=im.getpalette()
            im.putpalette(pal)
            nim=Image.new("RGB",im.size)
            nim.paste(im)
            ar=np.asarray(nim)
            nfi=str(i)+str(ext)
            fop=os.path.join(out,nfi)
            cv.imwrite(fop,ar[:,:,0])
            i+=1
rdgif('datadir/train/labels_gif','datadir/train/labels_jpg','.jpg')


# resize imgs and labels to 256 x 256 x 3 pixels
def rsiz(src):
    for i in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            im=Image.open(fp)
            width=256
            height=256
            rim=im.resize((width,height),Image.NEAREST)
            rim.save(fp)
rsiz('datadir/train/ims')
rsiz('datadir/train/labels_jpg')


# check image shape
im=cv.imread('datadir/train/ims/2.jpg')
im.shape
lab=cv.imread('datadir/train/labels_jpg/2.jpg')
lab.shape

