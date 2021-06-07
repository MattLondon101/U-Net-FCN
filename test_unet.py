# exec(open('imports.py').read())

import PIL
from natsort import natsorted,ns
import tensorflow as tf
tf.__version__
import time
mean=np.mean
import statistics
std=statistics.stdev
import importlib
from importlib import reload

#start_time=time.time()

import data
import model
import metrics

# exec(open('test_unet.py').read())

reload(data)
import data
from data import *
reload(model)
import model
from model import *
reload(metrics)
import metrics
from metrics import *

# tf.debugging.set_log_device_placement(True)


def testUnet1(lr,im_size,weights,imdir,numpreds,savedir,thresh):

    model = unet(pretrained_weights=None,input_size=im_size,learn_rate=lr)

    model.load_weights(weights)

    testGene = testGenerator(imdir,numpreds,target_size=im_size)

    results = model.predict_generator(testGene,numpreds,verbose=1)

    return saveResult(savedir,results,thresh)

weights='weights/run1_shuffle1.hdf5'
imdir='datadir/test/ims_cars_1'
savedir='datadir/test/predict_shuffle1_thresh50'
im_size=(256,256,3)
lr=1e-4
numpreds=20
thresh=0.40


testUnet1(lr,im_size,weights,imdir,numpreds,savedir,thresh)

# exec(open('test_unet.py').read())


## Make new directory for predictions
# os.mkdir('datadir/test/predict2')

## copy weights file
# shutil.copy('weights/run1_cars1.hdf5','weights/run1_cars2.hdf5')

## Clear scalars dir for tensorboard
# rm -R logs/scalars/*

# # rename predictions
# rim('data/Lab/Preds_Lab/pred15_19_GS','.png')
# invc('data/Home/Model_2__Subjs_Ims_Label_GS_Home/Subs_3_5_Ims_0_7/preds/CorrectPred2Sub3','data/Home/Model_2__Subjs_Ims_Label_GS_Home/Subs_3_5_Ims_0_7/preds/noninv_CorrectedPredSub3')

def ren(predir):
    i=0
    for im in natsorted(os.listdir(predir),alg=ns.PATH):
        fp=os.path.join(predir,im)
        if os.path.isfile(fp):
            d=str(i)+str('.jpg')
            fop=os.path.join(predir,d)
            os.rename(fp,fop)
            i+=1

ren(savedir)

# Get mean(std) jaccard score
def jacos(predir,labdir):
    jacs=jaccard_test(predir,labdir)
    # all_jacs=[round(i,4) for i in jacs]
    minvjacs=mean(jacs)
    stdinvjacs=std(jacs)
    rmj=round(minvjacs,4)
    rsj=round(stdinvjacs,4)
    print("mean (stdv) Jaccard score = " + str(rmj) + ' (' + str(rsj) + ')')

labdir='datadir/test/labels_cars_1'
jacos(savedir,labdir)


reload(data)
import data
from data import *
reload(model)
import model
from model import *
reload(metrics)
import metrics
from metrics import *

