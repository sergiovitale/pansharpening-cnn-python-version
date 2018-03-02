# pansharpening-cnn
Target-Adaptive CNN Based Pansharpening is an advanced version of pansharpening method PNN with residual learning, different loss and a target-adaptive phase.

# Team members
 Giuseppe Scarpa  (gi.scarpa@.unina.it)
 Sergio Vitale    (sergio.vitale@uniparthenope.it)
 Davide Cozzolino (davide.cozzolino@unina.it)
 
# License
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)

# Prerequisits
This code is written for Python2.7 and uses Theano library.
The list of all requirements is in "requirements.txt" file
The command to install the requirements is: 

`cat requirements.txt | xargs -n 1 -L 1 pip2 install`

Optional requirements for using gpu:
* cuda = 8
* cudnn = 5

# Usage
To use the code:
* set all paramaters in 'config_testing_#sensor.xml' for desired #sensor:
		*sensor: 	GeoEye1, IKONOS, WV2
		*mode: 		'full' to high resolution pansharpening (PAN scale)
				'reduce' to low resolution pansharening (MS scale)
		*fine tuning:
				set epochs '0' to not fine tune
				set epochs 'n' to do n epochs of fine tuning
		*path:	
				path of pretrained network
				path of image to test
				path of output image
				path of fine tuned network

* run test without gpu:
	`python PNN_testing.py -s SENSOR`
* run test with gpu:
	`[PATH] python PNN_testing.py -g -s SENSOR`
where SENSOR can be: 'GE1','IK','WV2'
where PATH is your local path of cuda

The code generates the:
 './outputs/#SENSOR/output.mat' as file result
 './outputs/#SENSOR/FT_network/PNN_model.mat' as fine tuned network, whether you choose finetuning


