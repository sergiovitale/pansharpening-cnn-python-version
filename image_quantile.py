# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""
import numpy as np

def image_quantile(img,p):
    Nk, Nr, Nc = img.shape
    y = np.zeros((Nk, np.size(p)))
    for index in range(Nk):
        y[index,:] = quantile( img[index,:,:], p )
    return y

def quantile(x, p):
    x = np.sort(x.flatten())
    p = np.maximum(np.floor(np.size(x)*p), 0).astype('int')
    y=x[p]
    return y
    
def image_stretch(img, th):
    img = np.double(img)
    if np.size(th)==2:
        img = (img-th[0])/(th[1]-th[0])
    else:
        Nk,Nr,Nc = img.shape
        for index in range(Nk):
            img[index,:,:] = (img[index,:,:]-th[index,0])/(th[index,1]-th[index,0])
        
    return img
