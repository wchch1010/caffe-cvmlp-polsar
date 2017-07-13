# caffe-cvmlp-polsar Caffe complex valued neural network for classification of PolSAR images

**This is a modified version of [Caffe](https://github.com/BVLC/caffe) used to achive goals set for master thesis work <!--[architecture](link)** -->

## Task of this thesis is to implement and investigate the Complex-Valued Convolutional Multi-layer Perception(CV-MLP) in order to solve classiffcation problem of PolSAR Images.
In other words given PolSAR image pixels that are represented as complex-valued scattering
vector our task is to automatically label regions on PolSAR image, for example: vegetation,
buildings, roads and more by mapping each image pixel to land type.
In order to train CV-MLP we're going to use ground truth data that was manually labeled
by humans.

### Dataset

 Manually labeled image "Oberpfaffenhofen" of size $1390 * 6640$ with 5 classes(city, field, forest, grassland, street).\\	 
 Training dataset generated using 4000 randomly selected pixels for each labeled class by taking region of $15 \times 15$ pixels.\\
 Total 20000 entries.
 Testing dataset contains same amount of image patches from remaining pixels after random selection for training.

### Net specification

TBD

## License

This extension to the Caffe library is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/
