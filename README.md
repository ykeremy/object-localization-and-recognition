# object-localization-and-recognition
Term project for CS484 Introduction to Computer Vision course in Bilkent University. 2019-2020 Fall Semester.

## TODO
- Write custom transformer to match a sample's size to 224x224 by 0 padding (and if necessary resizing)
- Apply feature extraction to the 224x224x3 images to obtain 2048-dim feature vectors using ResNet50.
- Training a neural network with ResNet features.
- Extract candidate windows using Selective Search region proposal algorithm as described in the project assignment.
- Classification and localization
- Calculate evaluation metrics
- Write the final report


## Division of work

1. __Beril__: Reading n images and their labels, then preprocessing the data. Extracting ResNet50 features.
    * Output: 2048-dimensional ResNet50 features (n, 2048)

2. __Kerem__: Training n images
    * Input: 2048-dimensional ResNet50 features (n, 2048)
    * Output: 10 predictions for each class for an image (n, 10)

3. __Salih__: Extracting candidate windows
    * Input: Test images (n, 224, 224, 3)
    * Output: Array of bounding boxes (candidate regions)
    
## A few details on implementation (things not to forget)
OpenCV Selective Search implementation is not part of OpenCV official distribution. It is part of OpenCV's extra modules which are not released as a part of official OpenCV distribution, called [opencv_contrib](https://github.com/opencv/opencv_contrib), which also includes the base modules.

Here is a [link](https://pypi.org/project/opencv-contrib-python/) with installation details.
