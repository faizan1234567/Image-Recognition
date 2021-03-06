# -*- coding: utf-8 -*-
"""C1_W4_Assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C1/C1/W4/assignment/C1_W4_Assignment.ipynb
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""**Things to Note:**
1. When coding the `class myCallback`, Python 3 will run into an error
```python
TypeError: '>' not supported between instances of 'NoneType' and 'float'
```
when using the code
```python
if(logs.get('accuracy')>0.99):
```

For Python 3, use the following equivalent code line

```python
if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
```

2. You can run the notebook using TensorFlow 2.5.0
"""

#!pip install tensorflow==2.5.0

# this is used in downloading data from the google drive
#!pip install gdown

"""Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.
"""

import tensorflow as tf
import os
import zipfile

!gdown --id 1NvV6VhmrfU8JDZNoEbwJxwx_6dhyN5bf #downlaoding the dataset

zip_ref = zipfile.ZipFile("./happy-or-sad.zip", 'r')
zip_ref.extractall("./h-or-s")
zip_ref.close()

def train_happy_sad_model():
    
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs ={}):
        if(logs.get('accuracy') is not None and logs.get('accuracy')>=0.999):
          print('\nReached desired accuracy! therefore cancelling training')
          self.model.stop_training = True
        
    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer = RMSprop(0.001),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy']
                 )
    
    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory('/content/h-or-s',
                                                        target_size =(150, 150),
                                                        shuffle = True,
                                                        class_mode = 'binary',
                                                        batch_size = 8
                                                        )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit(train_generator,
                        epochs = 20,
                        steps_per_epoch = 5,
                        verbose = 1,
                        callbacks = [callbacks]
                       )
    
    return (history.history['accuracy'][-1], model)

acc, model =train_happy_sad_model()

model.evaluate()

