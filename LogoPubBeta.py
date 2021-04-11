import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras import utils, layers, optimizers, models, metrics
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

import os

# target image size
n_pixlo = 100
n_pixla = 100

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   "./Photos",
#   validation_split=0.2,
#   subset="training",
#   image_size=(100, 100),
#   seed = 123,
#   batch_size=32)

# class_names = train_ds.class_names
# print(class_names)

# All images will be rescaled by 1./255
datagen = image.ImageDataGenerator(rescale=1./255, validation_split = 0.2) #Normalisation

# train set
train_generator = datagen.flow_from_directory("./Photos/TF1/Photos", subset='training', target_size=(n_pixlo,n_pixla), batch_size=32)

# validation set
valid_generator = datagen.flow_from_directory("./Photos/TF1/Photos", subset='validation', target_size=(n_pixlo,n_pixla), batch_size=32)

class_names = train_generator.classes
print(len(class_names))

classes = list(train_generator.class_indices.keys())
print(classes)

# Define the network
model = models.Sequential()
model.add(layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(n_pixlo,n_pixla,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.SeparableConv2D(64, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
#model.add(layers.Dense(1, activation='sigmoid')) #Logo ou pas logo
model.add(layers.Dense(2,activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

#path = "training/cp.ckpt"
#direction = os.path.dirname(path)

#mcp_save = ModelCheckpoint("./Modeles/test.h5", save_best_only=True, verbose = 1, monitor = "acc", mode = "auto")

# Train the network
history = model.fit(
      train_generator,
      steps_per_epoch=50,
      validation_data=valid_generator,
      epochs=20,
      )
      #callbacks = [mcp_save]
      #validation_steps=50)

# acc = history.history['acc']
# val_acc = history.history['val_acc']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(20)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

#model.save("param.h5", None)

#for key in history.history :
#	print(key)

# Get the class names from the training set
classes = list(train_generator.class_indices.keys())
print(classes)

# Read the test images
datagen = image.ImageDataGenerator(rescale=1./255) #Normalisation
test_generator = datagen.flow_from_directory("./Test", class_mode=None, target_size=(n_pixlo,n_pixla), batch_size=9)
test_images = next(test_generator)

# Test the network
preds  = model.predict(test_generator)
labels = np.argmax(preds, axis=1)
print(labels)

f, ax = plt.subplots(3, 3, figsize=(12,12))
for i, img in enumerate(test_images):
    ax.flat[i].imshow(img)
    ax.flat[i].axis("off")
    ax.flat[i].set_title(classes[labels[i]])
plt.show()