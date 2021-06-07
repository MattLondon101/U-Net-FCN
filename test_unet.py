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

dependencies = {
    'jaccard_index': jaccard_index
}

# tf.debugging.set_log_device_placement(True)

class testit():
    def __init__(self,wmethod,wpath,imsize,impath,labdir,predpath):
        self.wmethod=wmethod
        self.wpath=wpath
        self.impath=impath
        self.imsize=imsize
        self.predpath=predpath
        self.labdir=labdir
        self.numpreds=len(os.listdir(impath))
        self.thresh=0.5
        self.lr=1e-4

    def testunet1(self):
        if self.wmethod == 1:
            model = unet(pretrained_weights=None,input_size=self.imsize,learn_rate=lr)
            model.load_weights(self.wpath)
        elif self.wmethod == 2:
            yfi=open(self.wpath,'r')
            ly=yfi.read()
            yfi.close()
            model=model_from_yaml(ly)
        else:
            model = unet(pretrained_weights=None,input_size=self.imsize,learn_rate=lr)
            model.load_weights(self.wpath)


        testGene = testGenerator(self.impath,self.numpreds,target_size=self.imsize)
        results = model.predict_generator(testGene,self.numpreds,verbose=1)

        return saveResult(self.predpath,results,self.thresh)

    def ren(self):
        i=0
        for im in natsorted(os.listdir(self.predpath),alg=ns.PATH):
            fp=os.path.join(self.predpath,im)
            if os.path.isfile(fp):
                d=str(i)+str('.jpg')
                fop=os.path.join(self.predpath,d)
                os.rename(fp,fop)
                i+=1

    # Get mean(std) jaccard score
    def jacos(self):
        jacs=jaccard_test(self.predpath,self.labdir)
        # all_jacs=[round(i,4) for i in jacs]
        minvjacs=mean(jacs)
        stdinvjacs=std(jacs)
        rmj=round(minvjacs,4)
        rsj=round(stdinvjacs,4)
        print("mean (stdv) Jaccard score = " + str(rmj) + ' (' + str(rsj) + ')')

# User input commands
wmethod=int(input("There are two options for weights file format: 1. hdf5 and 2. yaml. Enter appropriate number to load from that format: "))
wpath=str(input("Enter path to weights file: "))
imsize=tuple(int(x.strip()) for x in input("Enter size of images as comma-separated integers (e.g. 256,256,3 for RGB images. 256,256 for grayscale): ").split(','))
# imsize=tuple(input("Enter size of images as comma-separated integers enclosed in parenthesis (e.g. (256,256,3) for RGB images. (256,256) for grayscale): "))
impath=str(input("Enter path to test images: "))
labdir=str(input("Enter path to test labels: "))
predpath=str(input("Enter path to directory to save prediction images: "))

# Initiate class
action=testit(wmethod,wpath,imsize,impath,labdir,predpath)
action.testunet1()
action.ren()
action.jacos()



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


# reload(data)
# import data
# from data import *
# reload(model)
# import model
# from model import *
# reload(metrics)
# import metrics
# from metrics import *
