# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""
"""
save all parameters loaded from .xml file in dictionaries:
save some parameters in variables to get suitable in next steps
 
-test_setting:          parameters for testing phase
-path:                  input and output path
-fine_tuning_setting:   parameters for fine tuning phase
"""



test_setting=model['test_setting']
path=model['path']
fine_tuning_setting=test_setting['fine_tuning_setting']

testset_path=path['testset_path']
test_dir_out=path['test_dir_out']
model_path=path['pretrained_model']

sensor = test_setting['sensor']
mode=test_setting['mode']

epochs=fine_tuning_setting['epochs']
ftnetwork_dir_out=path['ftnetwork_dir_out']
    
if test_setting.has_key('area'):
    area=test_setting['area']

PNN_model=sio.loadmat(model_path,squeeze_me=True)
residual = PNN_model['residual']

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

