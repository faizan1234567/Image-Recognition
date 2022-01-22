# -*- coding: utf-8 -*-
"""Untitled61.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ypXyOePlr_9yF8mV3p7MxZAN11-1XQNT
"""

# import all the necessary packages to implement transfer learning

import os
from tensorflow.keras import Model
from tensorflow.keras import layers

!wget --no-check-certificate \
https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
-O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3

pretrained_model = InceptionV3(input_shape = (150,150,3),
                               include_top = False,
                               weights = None)

weights = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model.load_weights(weights)

for layer in pretrained_model.layers: # freezing the early layers of the model
  layer.trainable = False

pretrained_model.summary()

last_layer = pretrained_model.get_layer('mixed7')

print('last layer shape:', last_layer.output.shape)

last_output = last_layer.output

# now stake the model to build the custom classification model using transfer learning
input = pretrained_model.input
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = input, outputs = output )

from tensorflow.keras.optimizers import RMSprop

# compile the model
model.compile(optimizer = RMSprop(learning_rate = 0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

!gdown --id 1RL0T7Rg4XqQNRCkjfnLo4goOJQ7XZro9

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

file = zipfile.ZipFile('./cats_and_dogs_filtered.zip', 'r')
file.extractall('/tmp')
file.close()

base_dir = '/tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')


validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')



print('number of cats images in the training directory:', len(os.listdir(train_cats_dir)))
print('number of dogs images in the training directory:', len(os.listdir(train_dogs_dir)))
print('number of cats images in the validation directory:', len(os.listdir(validation_cats_dir)))
print('number of dogs images in the validation directory:', len(os.listdir(validation_cats_dir)))

train_gen = ImageDataGenerator(rescale = 1/255, 
                               rotation_range = 40, 
                               width_shift_range= 0.2,
                               height_shift_range= 0.2,
                               shear_range= 0.2,
                               zoom_range= 0.2,
                               horizontal_flip= True,
                               fill_mode = 'nearest')

validation_gen = ImageDataGenerator(rescale = 1/255)

training_generator = train_gen.flow_from_directory(train_dir,
                                                   target_size = (150, 150),
                                                   batch_size = 20,
                                                   class_mode= 'binary')

validation_generator = validation_gen.flow_from_directory(validation_dir,
                                                          target_size = (150, 150),
                                                          batch_size = 20,
                                                          class_mode = 'binary')

#now, it's all set to train our model.

history = model.fit(training_generator,
                    validation_data= validation_generator,
                    epochs = 20,
                    steps_per_epoch = 100,
                    verbose = 2)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
