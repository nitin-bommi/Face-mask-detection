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

### Developing the model

To develop the model, my suggestion would be to gather more data from google images and manually select the portion of the face and categorise into respective classes. Then use your own architecture for developing.

The following model is developed with data inputs as images(.png) and annotations(.xml)
`preprocessing_dataset.py` is used to process the images and extract the faces and labels from annotations (xml parsing)
The images are the copied into separate folders with the class name as the folder name.
```python
face_img = img[y_min:y_max, x_min:x_max]

if name=='with_mask':
    cv2.imwrite(f'dataset\with_mask\image{m}.png', face_img)
    m += 1
```

While loading the data, it is first one hot encoded using `to_categorical()` and then split into train and test sets. 

Data augmentaion is used to increase the diversity of the data without gethering much data
```python
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
```
The parameters can be tuned for better results. 

Then the MobileNetV2 model is loaded from `tensorflow.keras.applications.MobileNetV2()` without the last layer by setting `include_top = False`. Once the base model is loaded, the head is architectured as above. 

The model is then comipled with:
+ loss = categorical_crossentropy
+ optimizer = Adam
+ metrics = accuracy

The face detection model is then loaded as
```python
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxtPath, weightsPath)
```

We now have the face detections and can use `cv2` to loop over the detections and mark the ROI with certain confidence level.

The same model is used for video predictions. 

*(Please raise an issue if something is not working)*
