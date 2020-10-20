# DeepVess
[***DeepVess***](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213539)  is a 3D CNN segmentation method with essential pre- and post-processing steps, to fully automate the vascular segmentation of 3D *in-vivo* MPM images of murine brain vasculature using [TensorFlow](https://github.com/tensorflow/tensorflow).

Additionally, The `topological loss` directory has the code and model related to the [***Topological Encoding CNN paper***](http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html).   

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

You can send me a sample image and I run DeepVess for you to see if this model works on your images.

## Requirements
* [Python 3](https://www.python.org) (It's compatible with Python 2 as well)
* [TensorFlow 1.14+](https://www.tensorflow.org) (With older version you may use [commit fee62a2](https://github.com/mhaft/DeepVess/tree/fee62a24ca2176027ab9d9c1c505f6340b59480d))
* [Matlab](https://www.mathworks.com) 
    * Image Processing Toolbox (if using motion removal in `prepareImage.m`)
    * Bioinformatics Toolbox (if using Fix the path of centerlines to have straight centerlines in `postProcess.m`)

## Publication
* Haft-Javaherian, M., Fang, L., Muse, V., Schaffer, C. B., Nishimura, N., & Sabuncu, M. R. (2019). Deep convolutional neural networks for segmenting 3D in vivo multiphoton images of vasculature in Alzheimer disease mouse models. PloS one, 14(3), e0213539. [Open Access link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213539).
* Haft-Javaherian, M., Villiger, M., Schaffer, C. B., Nishimura, N., Golland, P., & Bouma, B. E. (2020). A Topological Encoding Convolutional Neural Network for Segmentation of 3D Multiphoton Images of Brain Vasculature Using Persistent Homology. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 990-991). [Open Access link](http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html).
## Contact
* Mohammad Haft-Javaherian <mh973@cornell.edu>, <haft@csail.mit.edu>

## License
[Apache License 2.0](LICENSE)
