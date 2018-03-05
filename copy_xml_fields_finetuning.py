# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""
"""
load parameters from PNN_model(dictionary of parameters for training):
-pretrained_lr
-cost function
-regolarization term
-net_scope
-patch_size
"""

pretrained_lr=PNN_model['lr']
cost=PNN_model['cost']
regol=PNN_model['regol']

if PNN_model.has_key('net_scope'):
    net_scope=PNN_model['net_scope']
else:
    layers=[PNN_model['layers'][i] for i in xrange(0,len(PNN_model['layers']),2)]
    net_scope=0
    for lay in layers:
        net_scope+=lay.shape[2]-1
    net_scope=net_scope+1
    PNN_model['net_scope']=net_scope
    
if PNN_model.has_key('patch_size') :
    patch_size=PNN_model['patch_size']
else:
    patch_size=PNN_model['block_size']
    del PNN_model['block_size']
    PNN_model['patch_size']=patch_size
    


 
