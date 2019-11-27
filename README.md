# pansharpening-cnn-python-version
[Target-adaptive CNN-based pansharpening](https://ieeexplore.ieee.org/document/8334206) is an advanced version of pansharpening method [PNN](http://www.mdpi.com/2072-4292/8/7/594) with residual learning, different loss and a target-adaptive phase. 


# Team members
 Giuseppe Scarpa  (giscarpa@.unina.it);
 Sergio Vitale    (contact person, sergio.vitale@uniparthenope.it);
 Davide Cozzolino (davide.cozzolino@unina.it).
 
 
# License
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)

# Prerequisits
This code is written for Python2.7 and uses Theano library.
The list of all requirements is in `requirements.txt`.

The command to install the requirements is: 

```
cat requirements.txt | xargs -n 1 -L 1 pip2 install
```

Optional requirements for using gpu:
* cuda = 8 
* cudnn = 5

# Usage
* set all paramaters in `config_testing_<SENSOR>.xml`:
	* sensor:	GeoEye1, IKONOS, WV2
	* mode:		`full` to high resolution pansharpening (PAN scale); `reduce` to low resolution pansharening (MS scale).
	* fine tuning:	set epochs `0` to not fine tune; set epochs `n` to do n epochs of fine tuning.
	* paths:	path of pretrained network; path of image to test; path of output image; path of fine tuned network.

* run test without gpu:

```	
python PNN_testing.py -s <SENSOR>
```

* run test with gpu:

```
PATH=<CUDAPATH>:$PATH python PNN_testing.py -g -s <SENSOR>
```

* Output:
	* result: `./outputs/<SENSOR>/output.mat`;
	* (eventual) fine-tuned network: `./outputs/<SENSOR>/FT_network/PNN_model.mat`.
	

Where `<SENSOR>` can be: 'GE1','IK','WV2'; 
and `<CUDAPATH>` is your bin local path of cuda.
