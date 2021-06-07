def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            '''
            UNCOMMENT below if not binary classification (i.e. more classes than just 1. region of interest and 2. not region of interest) Note: Masks must be prepared to meet this need.
            '''
            # # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def newtraingen()



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict=dict(),image_color_mode = None,mask_color_mode = None,image_save_prefix  = "image",mask_save_prefix  = "mask",flag_multi_class = False,num_class = None,save_to_dir = None,target_size = None,shuffle=shuffle,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    data=[]
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        shuffle = shuffle,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        shuffle = shuffle,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

im_color='rgb'
mask_color='grayscale'
multi_class=False
num_class=1
shuffle=True
targ_size=(256,256)
im_size=(256,256,3)
root_datadir='datadir/train'
image_folder='ims_cars_num'
mask_folder='labels_cars_jpg_num'
# weights='weights/run1_cars1_1class1.hdf5'
pretrain=None
# pretrain=weights # if updating weights
batch_size=4
spe=40
epoc=3
lr=1e-4

lgen=trainGenerator(batch_size,root_datadir,image_folder,mask_folder,aug_dict=dict(),image_color_mode = im_color,mask_color_mode = mask_color,image_save_prefix  = "image",mask_save_prefix  = "mask",flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = targ_size,shuffle=shuffle,seed = 1)

trainGenerator(batch_size,root_datadir,image_folder,mask_folder,augs,weights,pretrain,im_color,mask_color,num_class,targ_size,im_size,lr,spe,epoc)


lgen=list(trainGenerator(batch_size,root_datadir,image_folder,mask_folder,aug_dict=dict(),image_color_mode = im_color,mask_color_mode = mask_color,image_save_prefix  = "image",mask_save_prefix  = "mask",flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = targ_size,shuffle=shuffle,seed = 1))
















/home/max/anaconda3/envs/unetenv/bin/pip install -r requirements.txt

rsiz('data/RGB/Lab/ims_lab_RGB_33')

def delet(src):
    for i,j,k in os.walk(src):
        for l in k:
            os.remove(os.path.join(i,l))
delet('data/RGB/Lab/Model_1__Subjs_Ims_Label_RGB_Lab')


src='data/Home/Model_2__Subjs_Ims_Label_GS_Home'
for i in natsorted(os.listdir(src),alg=ns.PATH):
    dp=os.path.join(src,i)
    os.mkdir(os.path.join(dp,'preds'))
    os.mkdir(os.path.join(dp,'weights'))


# os.walk: create/delete directories/files
src='data/GS/Lab/Model_1__Subjs_Ims_Label_GS_Lab'
for i,j,k in os.walk(src):
    # os.mkdir(os.path.join(i,'ims'))
    # os.mkdir(os.path.join(i,'inv_labels'))
    # os.mkdir(os.path.join(i,'labels'))
    # if os.path.basename(i)=='inv_labels':
    #     for l in natsorted(os.listdir(i),alg=ns.PATH):
    #         fp=os.path.join(i,l)
    #         if os.path.isfile(fp):
    #             os.remove(fp)
   

# move specific number of files to other dirs
def movfi(src,out,copyormv,nstart,start,stop):
    n=nstart
    for i in natsorted(os.listdir(src),alg=ns.PATH)[start:stop]:
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            np=os.path.join(out,str(n)+'.png')
            if str(copyormv)=='move':
                shutil.move(fp,np)
                pass
            elif str(copyormv)=='copy':
                shutil.copy(fp,np)
                pass
            else:
                print ('breaking')
                break
            n+=1
src='data/RGB/Lab/ims_lab_RGB_33'
out='data/RGB/Lab/Model_0__5_to_5_ims_RGB/ims_5_9_RGB_lab'
copyormv='copy'
nstart=0
start=5
stop=10
movfi(src,out,copyormv,nstart,start,stop)


def deleti(src):
    for i in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            os.remove(fp)
deleti('data/Home/Subjs_Ims_Label_GS_Home/Subs_5_9_Ims_3_10/labels')

# rename files in dirs
def rim(src,ext):
    i=0
    for im in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,im)
        if os.path.isfile(fp):
            d=str(i)+str(ext)
            fop=os.path.join(src,d)
            os.rename(fp,fop)
            i+=1

# rename multi dirs
def rendir(src,key,name):
    for i,j,k in os.walk(src):
        if os.path.basename(i)==key:
            fpa=os.path.split(i)
            os.rename(i,os.path.join(fpa[0],name))
src='data/Home/Subjs_Ims_Label_GS_Home'
rendir(src,'label','inv_labels')



# remove paths
src='data/Home/Subjs_Ims_Label_GS_Home'
for i in natsorted(os.listdir(src),alg=ns.PATH)[5:]:
    os.remove(os.path.join(src,i))

# Weights metrics
wei=model.get_weights()

for layer in model.layers:
    print(layer.get_config(), layer.get_weights())


# Spaces
from sys import getsizeof
gs=getsizeof
def sizit():
    lis=[]
    for i in globals():
        lis.append(gs(i))
    return lis
sizit()

gs(globals())

id(testims)
id(src)
ad=id(ims)
var=[x for x in globals().values() if id(x)==ad]



movit('/media/max/toshiba_ext/cvSOC/Subj_Dir/Home/Ims_Home_Num','/media/max/toshiba_ext/cvSOC/Subj_Dir/Home/Ims_Home_Num')
rim('/media/max/toshiba_ext/cvSOC/Subj_Dir/Home/Ims_Home_Num','.png')
shutil.copy('weights/5_to_5_Subjs_Lab_GS_Weights/Box_Subjs_1_5_GS.hdf5','weights/5_to_5_Subjs_Lab_GS_Weights/Box_Subjs_1_5_GS (3rd copy).hdf5')
shutil.copytree('origs_drafts/original_Images_ROIs/Origs_Home/Orig_Home_RGB_Ims','data/Home/Orig_Home_RGB_Ims')
rsiz('origs_drafts/original_Images_ROIs/Origs_Home/Orig_Home_RGB_Ims')

gpustat -cpi


# copy and move files to subs
import itertools
def grouper(S, n):
    iterator = iter(S)
    while True:
        items = list(itertools.islice(iterator, n))
        if len(items) == 0:
            break
        yield items

import glob, os, shutil
def cpit(dirdir,filedir,start,stop):
    dirs=natsorted(os.listdir(dirdir),alg=ns.PATH)[start:stop]
    fnames = natsorted(os.listdir(filedir),alg=ns.PATH)
    for dirz,fnamez in zip(dirs,grouper(fnames, 5)):
        dirpath=os.path.join(dirdir,dirz)
        n=0
        for fil in fnamez:
            fp=os.path.join(filedir,fil)
            fop=os.path.join(dirpath,fil)
            shutil.copy(fp,fop)
            fip=os.path.join(dirpath,str(n)+'.png')
            os.rename(fop,fip)
            n+=1

dirdir='data/Lab/Subjs_Ims_Label_GS'
filedir='data/Lab/Dir_Ims_Lab/gs_ims_33'
cpit(dirdir,filedir,None,None)

def deleter(src):
    for i,j,k in os.walk(src):
        if os.path.basename(i).startswith('inv'):
            for l in os.listdir(i):
                os.remove(os.path.join(i,l))
deleter('data/Lab/ims_labels_GS')

def delet(src):
    for i,j,k in os.walk(src):
        for l in k:
            os.remove(os.path.join(i,l))
delet('data/Lab/ims_labels_GS')


ts=cv.imread('data/Lab/ims33rgb/0.png')
tt=cv.imread('data/Lab/trainLab/ims33rgb/0.png',0)
ts.shape
tt.shape
l1=nu(ts)
l2=nu(tt)

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
Diff(l1,l1)

# tw=tt.astype('float32')
# tv=np.where(tw==138,0,255)
ret,thresh=cv.threshold(tt,139,255,cv.THRESH_BINARY)
imsh(thresh)
nu(thresh)

tz=255-tt
cv.imwrite('data/Home/inv_pred_3/3.png',tz)

inv('origs_drafts/original_Images_ROIs/Origs_Lab/Origs_Lab_Labels/Orig_Lab_Labels_33','data/Lab/inv_orig_lab_labels_33')

im1='data/Lab/Preds_Lab/Edited_Inv_Preds/Edited_PreInv_pred15_19_GS/0.png'
im2='data/Lab/inv_labels_33_thresh_5/0.png'
im1=cv.imread(im1)
im1=cv.imread(im1,0)
im2=cv.imread(im2)
nu=np.unique
nu(im1)
nu(im2)

src='data/Lab/Preds_Lab/Edited_Inv_Preds/Edited_PreInv_pred15_19_GS'
for i in natsorted(os.listdir(src),alg=ns.PATH):
    fp=os.path.join(src,i)
    if os.path.isfile(fp):
        im=cv.imread(fp,0)
        print(nu(im))


/home/max/anaconda3/envs/tfgpu15/lib/python3.6/site-packages/keras/engine/training.py

#from data.py
def adat(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            ## for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trgen(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adat(img,mask,flag_multi_class,num_class)
        # yield (img,mask)
        return (img,mask)


data_gen_args = dict()
mygen = trgen(3,'data/Lab','Ims_Lab_RGB_33','Inv_Labels_Lab_33_Thresh_5',data_gen_args,save_to_dir = None)


len(myGene)
mg0=myGene[0]
len(mg0[0])
imsh(mg0[0])

mg1=myGene[1]
imsh(mg1[0])


rm -R logs/scalars/*


src='data/Lab/labels'
for i in os.listdir(src):
    f=os.path.join(src,i)
    d="labels_"+i[4:]
    fop=os.path.join(src,d)
    os.rename(f,fop)

src='data/Lab/ims'
for i in os.listdir(src):
    print(i)

import io
Xtest,ytest=testGen('data/testHome/ims','data/testHome/validate',11)

target_size=(256,256)
flag_multi_class = True
testpath='data/Lab/Inv_Labels_Lab_33_Thresh_5'
xim = io.imread(os.path.join(testpath,"%d.png"%0),as_gray = False)
xim.shape
xim = xim / 255
xim.shape
xim = trans.resize(xim,target_size)
xim.shape
xim = np.reshape(xim,xim.shape+(1,)) if (not flag_multi_class) else xim
xim.shape
xim = np.reshape(xim,(1,)+xim.shape)
xim.shape


xim = io.imread(os.path.join(testpath,"%d.png"%i),as_gray = as_gray)
xim = xim / 255
xim = trans.resize(xim,target_size)
xim = np.reshape(xim,xim.shape+(1,)) if (not flag_multi_class) else xim
xim = np.reshape(xim,(1,)+xim.shape)


rim('data/testLab/chartRot','.png')

def bina(src,out,min):
    num=0
    for f in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,f)
        if os.path.isfile(fp):
            im=cv.imread(fp,0)
            ret,thresh=cv.threshold(im,min,255,cv.THRESH_BINARY)
            outer=os.path.join(out,str(num)+'.png')
            cv.imwrite(outer,thresh)
            num+=1

bina('data/Lab/testLab/preds2','data/Lab/testLab/preds2_thresh',243)
bina('data/Lab/testLab/vals','data/Lab/testLab/vals_thresh',16)

im=cv.imread('predict66_3264_2448/3_predict.png',0)
ret,thresh=cv.threshold(im,127,255,cv.THRESH_BINARY)
cv.imwrite('predict66_3264_2448_Threshed_127/3.png',thresh)


def tgen(test_path,num_image,target_size = (256,256,3),flag_multi_class = False,as_gray = True):
    imgs=[]
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        # img = img / 255
        img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        imgs.append(img)
    return imgs


def tgen(test_path,out,num_image,target_size = (256,256,3),flag_multi_class = False,as_gray = False):
    # imgs=[]
    j=0
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        # img = img / 255
        img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        # img = np.reshape(img,(1,)+img.shape)
        # imgs.append(img)
        nim=os.path.join(out,str(j)+'.png')
        cv.imwrite(nim,img)
        j=+1
    # return imgs


# move files
import shutil
source_dir = '/path/to/source_folder'
target_dir = '/path/to/dest_folder'
file_names = os.listdir(source_dir)
for file_name in file_names:
    shutil.move(os.path.join(source_dir, file_name), target_dir)

# delete all files
dir = 'data/trainLab/labels_33_auged1'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

def delet(src):
    for i,j,k in os.walk(src):
        for l in k:
            os.remove(os.path.join(i,l))
delet('data/Lab/ims_labels_GS')


conda install cudatoolkit=10.0.130
conda install cudnn=7.6.4=cuda10.0_0

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

64/8=8
512/8=64
256/8=32
128/8=16
1024/8=128
256/64=4
64/256=.25

expected conv2d_386 to have shape (256, 256, 1) but got array with shape (256, 256, 3)
conv2d_410 (after model=unet() again)
386,410,434,458=every 24
24/3=8

tv=tgen('data/Lab/testLab/vals_thresh',33)
vg=tv[0]
vg.shape

tp = tgen('data/Lab/testLab/preds2','data/Lab/testLab/preds2',33)
tp = tgen('data/Lab/testLab/preds2_thresh',33)
pg=tp[0]
pg.shape
tgg=pg/255

def uni(arr):
    uns=[]
    for i in arr:
        j=np.unique(i)
        uns.append(j)
    unz=np.unique(uns)
    return unz

vun=uni(vg)
vun.dtype
pun=uni(pg)
pun.dtype

def uno(arr):
    uns=[]
    for i in arr:
        uns.append(np.unique(i))
    return uns
viu=uno(vim)

viu=np.unique(vim)

pim=cv.imread('data/trainLab/labels_33/0.png')
pim.shape
pu=np.unique(pim)

vim=cv.imread('data/Lab/testLab/vals/0.png')
vim.shape
vu=np.unique(vim)



import h5py
f=h5py.File('weights/New_Box_ims10_14_GS.hdf5','r')
list(f.keys())
mw=f['model_weights']
ow=f['optimizer_weights']

mw.keys()


import h5py
import numpy as np
# https://stackoverflow.com/questions/61725716/is-there-a-way-to-save-each-hdf5-data-set-as-a-csv-column
def dump_calls2csv(name, node):    
    if isinstance(node, h5py.Dataset) and 'calls' in node.name :
       print ('visiting object:', node.name, ', exporting data to CSV')
       csvfname = node.name[1:].replace('/','_') +'.csv'
       arr = node[:]
       np.savetxt(csvfname, arr, fmt='%5d', delimiter=',')
with h5py.File('weights/BoxWeights_3_12_21.hdf5', 'r') as h5r :        
    h5r.visititems(dump_calls2csv)

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

f = h5py.File('weights/BoxWeights_3_12_21.hdf5','r')
f.visititems(print_attrs)


import tables as tb
import numpy as np

with tb.File('weights/BoxWeights_1_14_21.hdf5', 'r') as h5r :     
    for node in h5r.walk_nodes('/',classname='Leaf') :         
       print ('visiting object:', node._v_pathname, 'export data to CSV')
       csvfname = node._v_pathname[1:].replace('/','_') +'.csv'
       np.savetxt(csvfname, node.read(), fmt='%d', delimiter=',')


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

f = h5py.File('weights/BoxWeights_1_14_21.hdf5','r')
f.visititems(print_attrs)
f.visititems()


def print_attrs(name, obj):
    keys=[]
    vals=[]
    print(name)
    for key, val in obj.attrs.items():
        keys.append(key)
        vals.append(val)
    return (keys,vals)
f = h5py.File('weights/BoxWeights_3_12_21.hdf5','r')
attrs=f.visititems(print_attrs)
attrs


def serialize_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()

serialize_example(f)

na=np.array(mw)
no=np.array(ow)

mw1=f.get('model_weights')
ow1=f.get('optimizer_weights')

with h5py.File("weights/BoxWeights_3_12_21.hdf5","r") as f:
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])

list(f)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

import deepdish as dd
dd.io.save


input_shape=()

tp=1750
fn=609
fp=768
tn=62408

def acuracy(tp,fp,tn,fn):
    acy=(tp+fp)/(tp+fp+tn+fn)
    return acy
acy=acuracy(1750,768,62408,609)
acy

def f1s(tp,fp,tn,fn):
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    kep=keras.epsilon()
    f1sc=2*((prec*rec)/(prec+rec+kep))
    return f1sc
f1sc=f1s(1750,768,62408,609)
f1sc

def presion(tp,fp):
    kep=keras.epsilon()
    prec=tp/(tp+fp+kep)
    return prec
prec=presion(tp,fp)
prec

def recl(tp,fn):
    kep=keras.epsilon()
    recal=tp/(tp+fn+kep)
    return recal
recal=recl(tp,fn)
recal

# multiclass
from unet_funcs.data import *

edir='data/tlgt256/E'
def adjDat(src):
    for i in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,i)
        j=cv.imread(i)
        k=s

# make dirs
def mkdirs(src,out):
    lis=natsorted(os.listdir(src),alg=ns.PATH)
    for i,j in zip(natsorted(os.listdir(src),alg=ns.PATH),lis):
        fop=os.path.join(out,i)
        os.mkdir(fop)
mkdirs('data/Lab/ims_labels_RGB','data/Lab/ims_labels_GS')

# rename files in subdirectories 0:
def renz(src):
    for i,j,k in os.walk(src):
        m=0
        for l in k:
            os.rename(os.path.join(i,l), os.path.join(i,str(m)+'.png'))
            m+=1
renz('data/Lab/ims_labels_RGB')

# copy and move files in subs
def cpt(src,out):
    for i in range(0,len(os.listdir(src)),5):
        print(i)
        for i,j,k in os.walk(src):
        m=0
        for l in k:
src='data/Lab/Dir_Ims_Lab/gs_ims_33'

# copy and move files in subs
import itertools
def grouper(S, n):
    iterator = iter(S)
    while True:
        items = list(itertools.islice(iterator, n))
        if len(items) == 0:
            break
        yield items

import glob, os, shutil
fnames = natsorted(os.listdir(src),alg=ns.PATH)
for i, fnames in enumerate(grouper(fnames, 5)):
    print(i)
    fp=os.path.join(src,fnames[i])
    print(fp)
    os.mkdir(dirname)
    for fname in fnames:
        shutil.move(fname, dirname)


# rename ims w/ list
def riml(src,lis,out):
    for i,j in zip(natsorted(os.listdir(src),alg=ns.PATH),lis):
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            im=cv.imread(fp,0)
            imc=im.copy()
            fop=os.path.join(out,j)
            cv.imwrite(fop,imc)
src='data/Lab/ims_labels_RGB'
lis=natsorted(os.listdir('data/Lab/ims_labels_RGB'),alg=ns.PATH)
out='C:/Users/matth/Documents/SOC/UNet_2_17_21/UNet_Dir/data/train/labels'
riml(src,lis,out)

# make more and rename 
import shutil
def cop(src,ext,out,i):
    ext="."+str(ext)
    for z in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,z)
        if os.path.isfile(fp):
            if fp.endswith(ext):
                d=str(i)+ext
                fop=os.path.join(out,d)
                fc=shutil.copy(fp,fop)
                i+=1
src='data/Lab/ims_labels_GS/ims_5_9'
out='data/Lab/ims_labels_GS/ims_5_9'
cop(src,'png',out,20)

# resize ims
def rsiz(src):
    for i in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            im=Image.open(fp)
            width=256
            height=256
            rim=im.resize((width,height),Image.NEAREST)
            rim.save(fp)
rsiz('data/Lab/trainLab/ims33rgb')

gtd=os.listdir('TypesLabelsGT')

om=os.mkdir
for i in gtd:
    om(os.path.join('tlph256',i))

src='data/Home/Subjs_Ims_Label_GS_Home'
jds=[]
for i,j,k in os.walk(src):
    jds.append(j)

class dirds():
    def __init__(self):
        self.dirdo=[]
        self.dirda=[]
        return(None)

dira=[]
for i,j,k in os.walk(src):
    dirr=dirds()
    dire=dirr.dirdo
    dird=os.path.split(i)
    dir1=dird[1]
    dire.append(dir1)
    dira.append(dire)
diro=dira[1:]

# resize files in subdirectories
import os
def redi(src,out):
    for i,j,k in os.walk(src):
        for l in k:
            if l.endswith(".png"):
                fp=os.path.join(i,l)
                im=Image.open(fp)
                width=256
                height=256
                rim=im.resize((width,height),Image.NEAREST)
                fpa=os.path.split(i)
                lis=[out,fpa[1],l]
                dst=os.path.join(*lis)
                rim.save(dst)
src='TypesLabelsPH'
out='tlph256'
redi(src,out)

# rename files in subdirectories
import os
def rend(src,k1,k2):
    for i,j,k in os.walk(src):
        for l in k:
            if l.endswith(".roi"):
                os.rename(os.path.join(i,l), os.path.join(i,l.replace(str(k1), str(k2))))
rend('C:/Users/matth/Documents/SOC/UNet_2_17_21/Box/renaming/renamers','9289','9298')

# rename files of types
import os
def renf(src,k1,k2):
    for i in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            if str(k1) in i:
                os.rename(fp, os.path.join(src,i.replace(str(k1), str(k2))))
renf('TypesLabelsPH','_PH','')


def dirmk(lis,src):
    for i in lis:
        j=i[:-5]
        pa=os.path.join(src,'predict_'+j)
        os.mkdir(pa)
src='C:/Users/matth/Documents/SOC/UNet_2_17_21/Home_2_17_21/PredictionTests_2_17_21/pdtest1'
dirmk(plis,src)

pip install --user h5py==2.10.0 --force-reinstall --no-cache-dir


# Confusion Matrix Pipeline
def reader(src):
    li=[]
    for f in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,f)
        if os.path.isfile(fp):
            im=cv.imread(fp,0)
            ret,thresh=cv.threshold(im,127,255,cv.THRESH_BINARY)
            li.append(thresh)
    return li
src='data/Lab/Box_12_2/test_3_2/gt'
gt=reader(src)
src='data/Lab/Box_12_2/test_3_2/pl'
ph=reader(src)

def flat(ars):
    fl=[]
    for i in ars:
        j=i.flatten()
        fl.append(j)
    return fl
gtf=flat(gt)
phf=flat(ph)

len(gt[0])
len(ph[0])

set(gtf[0])
set(phf[0])

# from sklearn.metrics import confusion_matrix

def conf(lg,lp):
    cons=[]
    for i,j in zip(lg,lp):
        cons.append(confusion_matrix(i,j))
    return cons
conz=conf(gtf,phf)

dfc=pd.DataFrame(np.concatenate(conz))
dfc.to_csv('data/Lab/Box_12_2/test_3_2/box12.csv')

dfc=pd.DataFrame()
for i in conz:
    np.concatenate
    

cm0=confusion_matrix(phf[0],gtf[0])
cm0d=pd.DataFrame(cm0)
cm0d

# count uniques
(un,cnts)=np.unique(phf[0],return_counts=True)



def sums(src):
    s=[]
    for f in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,f)
        if os.path.isfile(fp):
            im=cv.imread(fp,0)
            fl=im.flatten()
            le=len(fl)
            s.append(le)
    return s
src='/home/max/aHome_2_13_21/groundTruthHome'
gtl=sums(src)

# each image 65536 pixels

import math
math.sqrt(16144233)


# ims to array
def reader(src):
    gt=[]
    for f in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,f)
        if os.path.isfile(fp):
            im=cv.imread(fp,0)
            gt.append(im)
    return gt
src='/home/max/aHome_2_13_21/predictHomeBin'
ph=reader(src)

ret,thresh=cv.threshold(ph[0],127,255,cv.THRESH_BINARY)

def uniq(ar):
    unis=[]
    for i in ar:
        uni=[]
        for j in i:
            un=set(j)
            uni.append(un)
        unis.append(uni)
    return unis
uniph=uniq(ph)



def gr(im):
    nim=[]
    for i in im:
        num=[]
        for j in i:
            if j == 0:
                num.append(127)
            else:
                num.append(j)
        nim.append(num)
    return nim

jim=gr(im)
ar=np.array(jim)
blur=cv.blur(ar,(8,8))
blu=np.uint8(blur)
cv.imwrite('/home/max/cvSOC/SOC_2021_Chris_Rafik/SOC_2021/Home_2_12_21/predictHome2/3.png',blu)

# Count 1's
lis=df.iloc[:,1:].values.tolist()
lis
len(lis[0])
x=1
def cntv(df,start_col,val):
    ns=[]
    x=val
    st=int(start_col)
    lis=df.iloc[:,st:].values.tolist()
    for i in lis:
        j=i.count(x)
        ns.append(j)
    ones=sum(ns)
    return ones
cnt=cntv(df,1,1)
cnt

import os
import cv2 as cv
import numpy as np

def blurr(impath,outpath,imnum,blrn):
    im=cv.imread(impath,0)
    nim=[]
    for i in im:
        num=[]
        for j in i:
            if j == 0:
                num.append(127)
            else:
                num.append(j)
        nim.append(num)
    ar=np.array(nim)
    rn=int(blrn)
    blur=cv.blur(ar,(rn,rn))
    blu=np.uint8(blur)
    outer=os.path.join(outpath,str(imnum)+'.png')
    cv.imwrite(outer,blu)

impath='/home/max/cvSOC/SOC_2021_Chris_Rafik/SOC_2021/Home_2_12_21/groundTruthHome/3.png'
outpath='/home/max/cvSOC/SOC_2021_Chris_Rafik/SOC_2021/Home_2_12_21/predictHome3'
imnum=3
blrn=10
blurr(impath,outpath,imnum,blrn)


import os
import sys
for p in sys.path:
    print(p)

# sys.path.append(os.path.join(os.path.dirname('/home/max/cvSOC/SOC_2021_Chris_Rafik/SOC_2021/unet_2021/trainUnet2.py'),os.path.pardir))

sys.path.append(os.path.dirname('/home/max/anaconda3/envs/flaskapp2/bin'))

sys.path.append(os.path.dirname('/home/max/cvSOC/SOC_2021_Chris_Rafik/SOC_2021/unet_2021'))
sys.path.append(os.path.dirname('/home/max/cvSOC/SOC_2021_Chris_Rafik/SOC_2021/unet_2021/trainUnet2.py'))


export PYTHONPATH='C:/Users/matth/Documents/SOC/UNet_2_17_21/unet_funcs_2021'
echo $PYTHONPATH

# rename ims natsorted
import os
from natsort import natsorted, ns
def renm(src,out):
    srcFiles=os.listdir(src)
    i=0
    for fileName in natsorted(os.listdir(src),alg=ns.PATH):
        filePath=os.path.join(src,fileName)
        if os.path.isfile(filePath):
            im=cv.imread(filePath)
            dst=str(i)+'.png'
            fileOutPath=os.path.join(out,dst)
            cv.imwrite(fileOutPath,im)
            i+=1

src='C:/Users/matth/Documents/SOC/UNet_2_17_21/UNet_Dir/data/Original_Images_ROIs/Box/rgb_numBoxRotated'
out='C:/Users/matth/Documents/SOC/UNet_2_17_21/UNet_Dir/data/Original_Images_ROIs/Box/rgb_numBox'
src='C:/Users/matth/Documents/SOC/UNet_2_17_21/UNet_Dir/data/Original_Images_ROIs/Box/roizips_numBoxOld'
out='C:/Users/matth/Documents/SOC/UNet_2_17_21/UNet_Dir/data/Original_Images_ROIs/Box/roizips_numBox'
renm(src,out)


import cv2 as cv
from natsort import natsorted, ns
import os
def greys(src,out):
    i=0
    for im in natsorted(os.listdir(src),alg=ns.PATH):
        filePath=os.path.join(src,im)
        if os.path.isfile(filePath):
            gs=cv.imread(filePath,0)
            dst=str(i)+'.png'
            fileOutPath=os.path.join(out,dst)
            cv.imwrite(fileOutPath,gs)
            i+=1

im=greys('Lab/Original_Images_ROIs/NewBox_2_28_21/ims_rgb_num','Lab/Original_Images_ROIs/NewBox_2_28_21/ims_gs_256x256')


# make more and rename batch
import shutil
def copb(src,ext,batches,out):
    i=0
    ext="."+str(ext)
    for batch in range(batches):
        for z in natsorted(os.listdir(src),alg=ns.PATH):
            fp=os.path.join(src,z)
            if os.path.isfile(fp):
                if fp.endswith(ext):
                    d=str(i)+ext
                    fop=os.path.join(out,d)
                    fc=shutil.copy(fp,fop)
                    i+=1
src='data/Lab/trainLab/labels33'
ext='png'
batches=10
out='data/Lab/trainLab/labels330'
copb(src,ext,batches,out)


os.listdir('C:/Users/matth/Documents/SOC/Home_2020/ims')

im.size

# rename files
import shutil
def rlab(src):
    for i in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,i)
        if os.path.isfile(fp):
            if fp.endswith('.png'):
                os.rename(fp, os.path.join(src,i.replace('.png','_GT.png')))
                
src='C:/Users/matth/Documents/SOC/UNet_2_17_21/UNet_Dir_TypeTest1/data/Original_Images_ROIs/HomeTypeTest1/TypeDir1/TypesLabelsGT_Orig'
rlab(src)

def grey(src,im,out):
    i=0
    for im in natsorted(os.listdir(src),alg=ns.PATH):
        fp=os.path.join(src,im)
        if os.path.isfile(fp):
            img=cv.imread(fp,0)
            cv.imwrite(out,img)
            im=Image.open(out)
            w=256
            h=256
            rim=im.resize((w,h),Image.NEAREST)
            rim.save(out)
        return rim
im=grey('Lab/Original_Images_ROIs/NewBox_2_28_21/ims_rgb_num','1170v04.png','C:/Users/matth/Documents/SOC/unet_multiseg_2_5_21/NewBox/ims/0.png')



# lis=[]
# for i in range(0,1):
#     for i,j,k in os.walk(src):
#         lis.append(j)


from keras.applications.vgg19 import VGG19
model=VGG19()
model.save('weights/vgg19_model_file_.h5')










import importlib
from importlib import reload
import vgg19_model
from vgg19_model import *
vgg19_model=VGG_19()
vgg19_model.save('weights/vgg19_model_file')











