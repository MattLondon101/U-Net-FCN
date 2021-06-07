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

# Use if performing image augmentations
# augs=dict(rotation_range=20,width_shift_range=(-20,+20),height_shift_range=(-20,+20),shear_range=0.2,zoom_range=0.1,fill_mode='nearest')
augs=dict()

def trainUnet1(batch_size,root_datadir,image_folder,mask_folder,augs,weights,pretrain,im_color,mask_color,num_class,targ_size,shuffle,im_size,lr,spe,epoc):

    gene=trainGenerator(batch_size=batch_size,train_path=root_datadir,image_folder=image_folder,mask_folder=mask_folder,aug_dict=augs,image_color_mode=im_color,mask_color_mode=mask_color,num_class=num_class,target_size=targ_size,shuffle=shuffle)

    model = unet(pretrained_weights=pretrain,input_size=im_size,learn_rate=lr)

    model_checkpoint = ModelCheckpoint(weights, monitor='loss',verbose=1, save_best_only=True)

    return (model.fit_generator(gene,steps_per_epoch=spe,epochs=epoc,callbacks=[model_checkpoint,tensorboard_callback],max_queue_size=1))

    print('Training tool {} seconds'.format(time.time()-start_time))

im_color='rgb'
mask_color='grayscale'
multi_class=False
num_class=None
shuffle=True
targ_size=(256,256)
im_size=(256,256,3)
root_datadir='datadir/train'
image_folder='ims_cars_num'
mask_folder='labels_cars_jpg_num'
weights='weights/run1_shuffle1.hdf5'
pretrain=None
# pretrain=weights # if updating weights
batch_size=4
spe=40
epoc=1
lr=1e-4

trainUnet1(batch_size,root_datadir,image_folder,mask_folder,augs,weights,pretrain,im_color,mask_color,num_class,targ_size,shuffle,im_size,lr,spe,epoc)

# exec(open('train_unet.py').read())


## Open tensorboard from Linux terminal
# tensorboard --logdir=logs/scalars/ --reload_multifile True --reload_interval 5
# tensorboard --logdir=logs/scalars/ --reload_multifile True --reload_interval 5 --debugger_port 6006



## copy weights file
# shutil.copy('weights/run1_cars1.hdf5','weights/run1_cars2.hdf5')

## Clear scalars dir for tensorboard from Linux terminal
# rm -R logs/scalars/*

