# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
"""
import theano
import theano.tensor as T
import numpy as np

class Network:
     """ define a netowrk given a list of layers """
     
     def __init__ (self,layer):
        self.layers = layer
        self.x = T.ftensor4('x')
         
        lay1 = self.x
        last = False
        for i in xrange(len(self.layers)):
            if i==len(self.layers) - 1:
                 last=True
            lay = self.layers[i]
            lay1 = lay.set_inpt(lay1,last)
        self.output = lay1
        
     def build(self,img):
         I_in=T.ftensor4('I_in')
         setnet= theano.function([I_in],self.output,givens={self.x :I_in})
         return setnet(img)
         
class ConvLayer:
    """ define a convolutional layer with given wegight and biases"""
    def __init__ (self,weigth,bias, filter_shape=None):
        
        self.w=theano.shared(np.asarray(weigth,dtype=theano.config.floatX),borrow=True)
        self.b=theano.shared(np.asarray(np.squeeze(bias),dtype=theano.config.floatX),borrow=True)      
        self.filter_shape=self.w.get_value().shape
        
        
    def set_inpt(self, inpt,last):
        self.last = last
        self.inpt = inpt
        conv_out = T.nnet.conv2d(self.inpt, self.w,
                        filter_shape=self.filter_shape, filter_flip=False)#use false to enable correlation with GPU
        #no Relu on last layer
        if self.last==False:
            self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output
