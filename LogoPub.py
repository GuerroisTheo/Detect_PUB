import numpy as np
import matplotlib.pyplot as plt

from keras import utils, layers, models, optimizers
from keras.preprocessing import image

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# target image size
n_pix = 40

# All images will be rescaled by 1./255
datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train set
train_generator = datagen.flow_from_directory("logos", subset='training', target_size=(n_pix,n_pix), batch_size=32)

# validation set
#valid_generator = datagen.flow_from_directory("logos", subset='validation', target_size=(n_pix,n_pix), batch_size=501)

# Define the network
model = models.Sequential()
model.add(layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(n_pix,n_pix,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(64, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(501, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #Logo ou pas logo
#model.add(layers.Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

# Train the network
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10)
      #validation_data=valid_generator,
      #validation_steps=50)

# Get the class names from the training set
classes = list(train_generator.class_indices.keys())

# Read the test images
test_generator = datagen.flow_from_directory("logos_test", class_mode=None, target_size=(n_pix,n_pix), batch_size=9)
test_images = next(test_generator)

# Test the network
preds  = model.predict(test_generator)
labels = np.argmax(preds, axis=1)


