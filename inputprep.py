# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""

import numpy as np
from others import interp23
import sys

def input_preparation(I_MS_LR,I_PAN,param):
    """pre-process input to make suitable tensor"""
  
    NDxI_LR = [];
        
    mav_value=2**(np.float32(param['L']))

    #compute radiometric indexes
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
           
    #interpolation
    if param['typeInterp']=='interp23tap':
        I_MS = interp23(I_MS_LR, param['ratio'])
        if len(NDxI_LR)!=0:
            NDxI = interp23(NDxI_LR, param['ratio'])
            print 'ok'
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
    
    return I_in
