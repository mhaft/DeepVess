# DeepVess
[***DeepVess with topology loss***](http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html) is a model based on ***DeepVess*** in addition to a topological loss term. 
Download the trained model file from this Google Drive URL: [model-epoch2000.pt](https://drive.google.com/file/d/1bRTBTkcCfSdEK4GhXbd7fgZ2Qa7uo3dL/view?usp=sharing).  

## Requirements
* [Python 3](https://www.python.org) 
* [PyTorch](https://pytorch.org/) 
* [TopologyLayer](https://github.com/bruel-gabrielsson/TopologyLayer)
* [Matlab](https://www.mathworks.com) 
    * Image Processing Toolbox (if using motion removal in `prepareImage.m`)
    * Bioinformatics Toolbox (if using Fix the path of centerlines to have straight centerlines in `postProcess.m`)

## Publication
* Haft-Javaherian, M., Villiger, M., Schaffer, C. B., Nishimura, N., Golland, P., & Bouma, B. E. (2020). A Topological Encoding Convolutional Neural Network for Segmentation of 3D Multiphoton Images of Brain Vasculature Using Persistent Homology. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 990-991). [Open Access link](http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html).
## Contact
* Mohammad Haft-Javaherian <mh973@cornell.edu>, <haft@csial.mit.edu>

## License
[Apache License 2.0](../LICENSE)
