"""All imports necessary to carry out this process"""
import numpy as np
import matplotlib.pyplot as plt
from keras import utils, layers, optimizers, models
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

import os

# target image size
n_pixlo = 40
n_pixla = 40

# All images will be rescaled by 1./255
datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train set
train_generator = datagen.flow_from_directory("./Photos", subset='training', target_size=(n_pixlo,n_pixla), batch_size=32)

# validation set
#valid_generator = datagen.flow_from_directory("logos", subset='validation', target_size=(n_pix,n_pix), batch_size=501)

# Define the network
model = models.Sequential()
model.add(layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(n_pixlo,n_pixla,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(64, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(501, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #Logo ou pas logo
model.add(layers.Dense(2,activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

#path = "training/cp.ckpt"
#direction = os.path.dirname(path)

#mcp_save = ModelCheckpoint("test.h5", save_best_only=True, verbose = 1, monitor = "acc", mode = "auto")

# Train the network
history = model.fit(
      train_generator,
      steps_per_epoch=125,
      epochs=10,
      )
      #callbacks = [mcp_save]
      #validation_data=valid_generator,
      #validation_steps=50)

model.save("param.h5", None)

#for key in history.history :
#	print(key)

# Get the class names from the training set
classes = list(train_generator.class_indices.keys())
print(classes)

# Read the test images
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


