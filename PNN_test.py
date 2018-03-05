# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""
    
import numpy as np
import sys
from others import interp23
from fine_tuning import fine_tuning
import scipy.io as sio
from downgrade_images import downgrade_images

def PNN_test (I_MS_LR,I_PAN,inputImg, param,net,path,mode,epochs=0):
	 
    test_dir_out=path['test_dir_out']
    FTnetwork_dir_out=path['ftnetwork_dir_out']
    param['L']=inputImg['L']
    param['ratio']=inputImg['ratio']
    if 'inputType' not in param.keys():
        param['inputType']='MS_PAN'
    

    #fine tuning
    if epochs != 0:
        fine_tuning(I_MS_LR,I_PAN,param,epochs,FTnetwork_dir_out)        
        ft_model_path = FTnetwork_dir_out+'/PNN_model.mat'
        
        FT_model = sio.loadmat(ft_model_path,squeeze_me=True)
        
        from PNN_testing_model import Network, ConvLayer
        
        layer=[]
        for j in range(0,len(FT_model['layers']),2):
            layer.append(ConvLayer(FT_model['layers'][j], FT_model['layers'][j+1]))
        net=Network(layer)
                 
    
    if mode != 'full':
        I_MS_LR,I_PAN=downgrade_images(I_MS_LR,I_PAN,param['ratio'],param['sensor'])    
    
    I_PAN = np.expand_dims(I_PAN,axis=0)    
    NDxI_LR = [];    
    mav_value=2**(np.float32(param['L']))
    
    # compute radiometric indexes
    if param['inputType']=='MS_PAN_NDxI':
        if I_MS_LR.shape[0] == 8:
            NDxI_LR = np.stack((
                  (I_MS_LR[4,:,:]-I_MS_LR[7,:,:])/(I_MS_LR[4,:,:]+I_MS_LR[7,:,:]),
                  (I_MS_LR[0,:,:]-I_MS_LR[7,:,:])/(I_MS_LR[0,:,:]+I_MS_LR[7,:,:]),
                  (I_MS_LR[2,:,:]-I_MS_LR[3,:,:])/(I_MS_LR[2,:,:]+I_MS_LR[3,:,:]),
                  (I_MS_LR[5,:,:]-I_MS_LR[0,:,:])/(I_MS_LR[5,:,:]+I_MS_LR[0,:,:])),axis=0 )
        else:
            NDxI_LR = np.stack((
                                (I_MS_LR[3,:,:]-I_MS_LR[2,:,:])/(I_MS_LR[3,:,:]+I_MS_LR[2,:,:]),
                                (I_MS_LR[1,:,:]-I_MS_LR[3,:,:])/(I_MS_LR[1,:,:]+I_MS_LR[3,:,:])), axis=0 )
           
    #input preparation      
    if param['typeInterp']=='interp23tap':
        I_MS = interp23(I_MS_LR, param['ratio'])
        if len(NDxI_LR)!=0:
            NDxI = interp23(NDxI_LR, param['ratio'])
    else:
        sys.exit('interpolation not supported')
    
    
    if param['inputType']=='MS':
        I_in = I_MS.astype('single')/mav_value
    elif param['inputType']=='MS_PAN':     
        I_in = np.vstack((I_MS, I_PAN)).astype('single')/mav_value
    elif param['inputType']=='MS_PAN_NDxI':
        I_in = np.vstack((I_MS, I_PAN)).astype('single')/mav_value
        I_in = np.vstack((I_in, NDxI)).astype('single')
    else:
        sys.exit('Configuration not supported')
    print I_in.shape 

    I_in_residual=np.expand_dims(I_in,axis=0)
    I_in_residual=I_in_residual[:,:I_MS.shape[0],:,:]
    I_in = np.pad(I_in, ((0,0),(param['padSize']/2,param['padSize']/2),(param['padSize']/2,param['padSize']/2)),mode='edge')
    I_in = np.expand_dims(I_in,axis=0)
    
    #Pansharpening
    if param['residual']:
        I_out=net.build(I_in)+I_in_residual[:,:I_MS.shape[0],:,:]
    else:
        I_out=net.build(I_in)
        
    I_out = I_out * mav_value
    
    return np.squeeze(I_out)
    
    
