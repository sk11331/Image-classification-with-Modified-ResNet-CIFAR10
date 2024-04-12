# Image-classification-with-Modified-ResNet-CIFAR10
This is a machine deep learning model that trains on CIFAR-10 dataset for image classification
CS-GY 6953 / ECE-GY 7123 Deep Learning Mini-Project Spring 2024
New York University (NYU) Tandon School of Engineering
This repository hosts an implementation of a deep learning model for image classification utilizing a Modified ResNet Architecture. The primary objective is to enhance performance on a CIFAR-10 dataset while adhering to the constraint of maintaining the model's trainable parameters under 5 million.
# Overview
The aim of this mini-project is to create and train a convolutional neural network (CNN) model to accurately classify images from the CIFAR-10 dataset, aiming for a test accuracy of over 90%. The CIFAR-10 dataset comprises 60,000 color images of size 32x32, distributed across 10 classes, with 6,000 images per class.
# Model Architecture
The model architecture is derived from the ResNet (Residual Network) framework, which employs residual blocks to aid in training extremely deep networks. This ResNet architecture has been adapted to meet the particular demands of the CIFAR-10 dataset. It encompasses numerous convolutional layers, batch normalization layers, and residual blocks. The ultimate layer is a fully connected layer utilizing softmax activation to produce class probabilities.
# Training Methodology
The model underwent training utilizing the PyTorch deep learning framework on the CUDA platform. The training regimen encompassed multiple epochs, optimizing batch-wise using the Adam optimizer. To enhance the diversity of the training dataset and bolster generalization performance, data augmentation techniques like random flips and random crops were applied.
# Performance Evaluation
The trained model's performance was assessed on a distinct test dataset comprising previously unseen images from the CIFAR-10 dataset. Evaluation metrics included test accuracy and test loss.

The model achieved a final test accuracy of 92.67%, with only 2,777,674 trainable parameters.
# References
1 Liu, W. et al. (2021). Improvement of CIFAR-10 Image Classification Based on Modified ResNet-34. In: Meng, H., Lei, T., Li, M., Li, K., Xiong, N., Wang, L. (eds) Advances in Natural Computation, Fuzzy Systems and Knowledge Discovery. ICNC-FSKD 2020. Lecture Notes on Data Engineering and Communications Technologies, vol 88. Springer, Cham.
# 
2 Shuvam Das (2023). Implementation of ResNet Architecture for CIFAR-10 and CIFAR-100 Datasets
# 
3 Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian SunDeep Residual Learning for Image Recognition. arXiv:1512.03385
# Team Members
Shanmukeshwar Reddy Kanjula(sk11331)
Nathaniel Sehati(nys2021)
