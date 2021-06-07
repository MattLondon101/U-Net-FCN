exec(open('imports.py').read())

import tensorflow as tf
from tensorflow import keras
import time
start_time=time.time()
import importlib
from datetime import datetime
# from packaging import version


from importlib import reload
import data
import model
import metrics

reload(data)
import data
from data import *
reload(model)
import model
from model import *
reload(metrics)
import metrics
from metrics import *

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,profile_batch=0)


class trainit():
    def __init__(self,rootdir,imdir,labpath,augs,wdir,pretrain,imcolor,labcolor,shuffle,batchsize,spe,epoc,lr):
        self.rootdir=rootdir
        self.imdir=imdir
        self.labpath=labpath
        self.augs=augs
        self.wdir=wdir
        self.pretrain=pretrain
        self.imcolor=imcolor
        self.labcolor=labcolor
        self.numclass=None
        self.targsize=(256,256)
        self.shuffle=shuffle
        self.batchsize=batchsize
        self.spe=spe
        self.epoc=epoc
        self.lr=lr


    def trainUnet1(self):
        if self.imcolor == 1:
            imgcolor='rgb'
            imsize=(256,256,3)
        elif self.imcolor == 2:
            imgcolor='grayscale'
            imsize=(256,256)
        else:
            imgcolor='rgb'
        
        if self.labcolor == 1:
            maskcolor='rgb'
        elif self.labcolor == 2:
            maskcolor='grayscale'
        else:
            maskcolor='grayscale'

        if self.augs == 1:
            aug=dict()
        elif self.augs == 2:
            aug=dict(rotation_range=20,width_shift_range=(-20,+20),height_shift_range=(-20,+20),shear_range=0.2,zoom_range=0.1,fill_mode='nearest')
        else:
            aug=dict()
        
        gene=trainGenerator(batch_size=self.batchsize,train_path=self.rootdir,image_folder=self.imdir,mask_folder=self.labpath,aug_dict=aug,image_color_mode=imgcolor,mask_color_mode=maskcolor,num_class=self.numclass,target_size=self.targsize,shuffle=self.shuffle)

        if self.pretrain == 1:
            ptrain=self.wdir
        elif self.pretrain == 2:
            ptrain=None

        model = unet(pretrained_weights=ptrain,input_size=imsize,learn_rate=lr)

        model_checkpoint = ModelCheckpoint(self.wdir, monitor='loss',verbose=1, save_best_only=True)

        return (model.fit_generator(gene,steps_per_epoch=spe,epochs=epoc,callbacks=[model_checkpoint,tensorboard_callback],max_queue_size=1))

        print('Training tool {} seconds'.format(time.time()-start_time))



# User input commands
rootdir=str(input("Enter path to root directory of images and labels: "))
imdir=str(input("Enter name train image directory: "))
labpath=str(input("Enter name train label directory: "))
augs=int(input("Enter 1 for no image augmentation during training, else enter 2 to perform augmentation: "))
wdir=str(input("Enter path to save and update weights file: "))
pretrain=int(input("If updating model to pre-existing weights in path, enter 1. Else, if creating new weights file in path, enter 2: "))
imcolor=int(input("For image color, enter 1 for rgb or 2 for grayscale: "))
labcolor=str(input("For label color, enter 1 for rgb or 2 grayscale/binary: "))
shuffle=str(input("Enter True to shuffle training image, else False: "))
batchsize=int(input("Enter batch size: "))
spe=int(input("Enter steps-per-epoch: "))
epoc=int(input("Enter number epochs: "))
lr=float(input("Enter learning-rate as float (e.g. 0.0001): "))

# Initiate class
action=trainit(rootdir,imdir,labpath,augs,wdir,pretrain,imcolor,labcolor,shuffle,batchsize,spe,epoc,lr)

action.trainUnet1()



# exec(open('unet_train.py').read())

## Open tensorboard from Linux terminal
# tensorboard --logdir=logs/scalars/ --reload_multifile True --reload_interval 5
# tensorboard --logdir=logs/scalars/ --reload_multifile True --reload_interval 5 --debugger_port 6006



## copy weights file
# shutil.copy('weights/model1.hdf5','weights/model2.hdf5')

## Clear scalars dir for tensorboard from Linux terminal
# rm -R logs/scalars/*

