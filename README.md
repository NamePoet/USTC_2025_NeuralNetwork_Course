# USTC-2025-NeuralNetwork-Course                                              
Yilin YANG's course assignments of Neural Network.                                           







## Getting Started                
These projects all use PyTorch framework, we need to install PyTorch (appropriate version according to our CUDA version) in our Pycharm.  


[PyTorch Download](https://pytorch.org/)            


```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Project 1: Regression Analysis                  
### Boston House Price Forecast  

Dataset: Boston house price dataset, which has a dimension of (506,14), including 506 data,
and each data contains 14 feature dimensions. The characteristic dimension includes 
13 dimensions and the corresponding house price.  
![](https://pic1.imgdb.cn/item/67e0091488c538a9b5c51edb.png)


## Project 2：CNN CIFAR-10 Image Classification    

Dataset：The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

[![CIFAR-10.png](https://pic1.imgdb.cn/item/68032d9658cb8da5c8b47a75.jpg)](https://pic1.imgdb.cn/item/68032d9658cb8da5c8b47a75.jpg)

![](https://pic1.imgdb.cn/item/68032b5158cb8da5c8b46f33.png)

![](https://pic1.imgdb.cn/item/68032b6d58cb8da5c8b46fc3.png)

[![image.png](https://pic1.imgdb.cn/item/68032e6058cb8da5c8b47d8a.png)](https://pic1.imgdb.cn/item/68032e6058cb8da5c8b47d8a.png)

## Project 3:  FGSM Attack on the Facenet Model

Dataset: Welcome to Labeled Faces in the Wild, a database of face photographs designed for studying the problem of unconstrained face recognition. The data set contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. 1680 of the people pictured have two or more distinct photos in the data set. The only constraint on these faces is that they were detected by the Viola-Jones face detector.

![](https://pic1.imgdb.cn/item/680b922e58cb8da5c8ccf963.png)




