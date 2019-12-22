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