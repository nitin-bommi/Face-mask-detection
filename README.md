# Face-mask-detection

### Introduction

__Taking precautions has become so important during this qurantine and wearing a mask😷 is an important one.__

This model predicts 3 classes:
+ *A person wore a mask*
+ *A person did not wear a mask*
+ *A person wore a mask improperly*

With this last class, we can either warn them or teach them the proper way to wear a mask. 

We have developed code to predict from an image as well as a video.

### Model's architecture

The base architecture used is `MobileNetV2: Inverted Residuals and Linear Bottlenecks`. More about this architecture can be found [here](https://arxiv.org/abs/1801.04381). 

The head of the model is left untrained. The head is connected to a layer with 128 neurons and then a layer with 3 neurons as the output layer.

Additionally, a dropout of 0.5 is used as a good regularisation metric, flattening and pooling has also been performed for computational efficiency.

__The final architecture:__
+ A base MobileNetV2 layer (without ouput layer)
+ Average Pooling of pool_size (7 x 7)
+ Flattening the Network into a single dimentions to pass on to fully connected layers.
+ A network with 128 neurons
+ Dropout to reduce overfitting
+ The final layer with 3 neurons (3 classes)

### Model's performance

The model is trained on just 800 images (somewhat like a skewed dataset as the number of images for improperly worn class is very less) 

Since we imported the weights of the MobileNetV2 model, the model was able to do better. We are trying to gather more data from different sources. 

### Test the model

To run the model in your PC/laptop, follow the steps

+ Clone the repository
+ In your system, change your directory to Mask Detection
+ To train the data, add images to `dataset/{class_name}` and change the model, if any, and run the script as 
```bash
$ python train_mask_detector.py
```
If you are using some IDE, you can directly run the script from the run button.
+ Once you train the model, the model is saved in h5 format and can be used for later purpose.
+ For image prediction, you can upload an image in the same file and see the output.
+ For video, run the command
```bash
$ python test_mask_video.py
```
Again, if you are using some IDE, you can run the script directly.

