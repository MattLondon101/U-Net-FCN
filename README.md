# U-Net-FCN  
Pipeline for U-Net FCN with Keras and Tensorflow backend  

U-Net is an FCN that is popular for biomedical image segmentation. This repository includes the pipeline for preprocessing, training, and testing images.  

## Dependencies  
This was successfully run in a Conda environment with Python 3.7  
Commands below are meant to be used in Linux terminal.  

Create environment from Conda Base environment:  
```
cd U-Net_FCN
conda create -n unetenv python==3.7
```  

Install pip in conda environment:  
```
conda install pip
```  


Install requirements:  
```
python imports.py
```  

Open Tensorboard to monitor loss and Jaccard coefficient across epochs:  
```
tensorboard --logdir=logs/scalars/ --reload_multifile True --reload_interval 5
```  
  

<p align="center">
U-Net FCN Architecture
</p>
<p align="center">
  <img width="870" height="504" src="https://github.com/MattLondon101/U-Net-FCN/blob/main/u-net-architecture.png?raw=true"
</p>

