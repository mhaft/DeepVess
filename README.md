# DeepVess
[***DeepVess***](https://arxiv.org/abs/1801.00880)  is a 3D CNN segmentation method with essential pre- and post-processing steps, to fully automate the vascular segmentation of 3D *in-vivo* MPM images of murine brain vasculature using [TensorFlow](https://github.com/tensorflow/tensorflow). 

### How to use DeepVess

First, see [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions on how to install TensorFlow.

Second, run *prepareImage* in MATLAB. (See `Help prepareImage`)
```matlab
>> prepareImage()
```

Third, run *DeepVess* in Terminal or Python. You can add the address of the output of prepareImage (e.g. ../image3D.h5) as the argument. Otherwise, code will ask you to input it later.
```shell 
$ python DeepVess.py ../image3D.h5
```
Finally, run *postProcess* in MATLAB.
```matlab
>> postProcess()
```
Note that *prepareImage* and *postProcess* accepts arguments to avoid input request. For more information look at their helps in MATLAB.
 ```matlab
>> help prepareImage
>> help postProcess
```

## Publication
* Haft-Javaherian, M; Fang, L.; Muse, V.; Schaffer, C.B.; Nishimura, N.; & Sabuncu, M. R. (2018) Deep convolutional neural networks for segmenting 3D in vivo multiphoton images of vasculature in Alzheimer disease mouse models. *arXiv preprint, arXiv*:1801.00880.

## Contact
* Mohammad Haft-Javaherian <mh973@cornell.edu>

## License
[Apache License 2.0](LICENSE)
