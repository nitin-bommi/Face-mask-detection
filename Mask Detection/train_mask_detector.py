# Importing the libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Initialising some variables
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Loading the images
imagePaths = list(paths.list_images("dataset"))
data = []
labels = []

# Storing the images and the labels
for imagePath in imagePaths:
    
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

# Converting the data and labels to NumPy arrays for easy computation
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Performing one-hot encoding on the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Splitting the dataset into training and testing sets
(X_train, X_test, y_train, y_test) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=0)

# Using data augmentation for more data
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

datagen.fit(X_train)

# Using the base imagenet model and modifying the final layer
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_shape=(224, 224, 3))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# Compiling the model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# Training the model's last layer
model.fit(datagen.flow(X_train, y_train, batch_size=BS),
                    steps_per_epoch=len(X_train) // BS,
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) // BS,
                    epochs=EPOCHS)

# Evaluating model's performance
predIdxs = model.predict(X_test, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

# Printing the overall performance report
print(classification_report(y_test.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

# Saving the model for future use
model.save('model', save_format="h5")

# Defining the confidence level (for high precision, use ~0.8)
confidence_arg = 0.6

# Loading the face detector model
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Reading the image
image = cv2.imread('images/pic1.jpeg')
orig = image.copy()
(h, w) = image.shape[:2]

# Constructing a blob
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
    (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

# Detecting
for i in range(0, detections.shape[2]):
	
	# Probability that it is a face
    confidence = detections[0, 0, i, 2]

    if confidence > confidence_arg:
		
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# ROI, BGR->RGB
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

		# Using the model to predict 
        (noproperMask, mask, withoutMask) = model.predict(face)[0]

        if (mask>withoutMask and mask>noproperMask):
            label = "Mask"
            color = (0, 255, 0)
        elif (withoutMask>mask and withoutMask>noproperMask):
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "No proper mask"
            color = (255, 0, 0)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Output image with predictions
cv2.imshow("Output", image)
cv2.waitKey(0)
