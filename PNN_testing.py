# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser( 
        description = 'Target-Adaptive CNN Based Pansharpening')
                        
parser.add_argument('-g', '--gpu', action='store_true',default=False,
                        help='the identifier of the used GPU.')

parser.add_argument('-s', '--sensor', type=str, default='GE1',
                        help='the identifer of the used sensor.')

config, _ = parser.parse_known_args()
sensor = config.sensor                        
if (config.gpu):
	os.environ["THEANO_FLAGS"]='device=gpu0,floatX=float32,init_gpu_device=gpu0'
else:
	os.environ["THEANO_FLAGS"] = "floatX=float32"

import scipy.io as sio
from PNN_testing_model import Network, ConvLayer
from PNN_test import PNN_test
from others import parser_xml, export2

model=parser_xml('config_testing_'+sensor+'.xml')
execfile('copy_xml_fields_testing.py')

layer=[]
for i in xrange(0,len(PNN_model['layers']),2):
    layer.append(ConvLayer(PNN_model['layers'][i], PNN_model['layers'][i+1]))
net=Network(layer)

#%% Pansharpening

#load images     
inputImg=sio.loadmat(testset_path)
I_MS_LR = np.array(inputImg['I_MS'],dtype='double').transpose(2,0,1)
I_PAN = np.array(inputImg['I_PAN'],dtype='double')

#Testing
I_MS_HR = PNN_test(I_MS_LR,I_PAN,inputImg, PNN_model,net,path,mode,epochs)
    
#%% save data
export2(I_MS_HR,test_dir_out)

#%% Visualization

from image_quantile import image_quantile, image_stretch
import matplotlib.pyplot as plt
plt.close('all')

I_PAN=np.expand_dims(I_PAN,axis=0)
plt.figure()
plt.subplot(131)
th_PAN = image_quantile(I_PAN, np.array([0.01, 0.99]))
PAN = image_stretch(np.squeeze(I_PAN),np.squeeze(th_PAN))
plt.imshow( image_stretch(np.squeeze(I_PAN),np.squeeze(th_PAN)),cmap='gray',clim=[0,1])
plt.title('PANCHROMATIC'), plt.axis('off')
    
RGB_indexes = np.array(inputImg['RGB_indexes'])
RGB_indexes = RGB_indexes - 1
    
plt.subplot(132)
th_MSrgb = image_quantile(np.squeeze(I_MS_LR[RGB_indexes,:,:]), np.array([0.01, 0.99]));
d=image_stretch(np.squeeze(I_MS_LR[RGB_indexes,:,:]),th_MSrgb)
d[d<0]=0
d[d>1]=1
plt.imshow(d.transpose(1,2,0))
plt.title('MULTISPECTRAL LOW RESOLUTION'), plt.axis('off')

plt.subplot(133)
I_MS_HR = np.squeeze(I_MS_HR)
c=image_stretch(np.squeeze(I_MS_HR[RGB_indexes,:,:]),th_MSrgb)
c[c<0]=0
c[c>1]=1
plt.imshow(c.transpose(1,2,0))
plt.title('MULTISPECTRAL HIGH RESOLUTION'), plt.axis('off')

plt.show()
