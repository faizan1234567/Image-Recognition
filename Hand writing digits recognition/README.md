Handwritten digits dataset was obtained from [MNIST](http://yann.lecun.com/exdb/mnist/) datasets, the dataset contained 60000 images for training and 10000 images for the test. A custom convolutional neural network was trained to recognize handwritten digits. The model is pretty simple, it consists of one [convolution layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D),  one [max-pooling layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D), and two [fully connected layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense). The number of classes, in this case, is 10 ( for 0-9 digits).  The optimizer for this problem is [Adam]( with sparse categorical cross-entropy loss.  The model has been trained for 20 epochs on GPU on [google colab](https://colab.research.google.com/). The model achieved 98.5 test accuracy, and it was correctly recognizing handwritten digits. 

The model achieved 98.5% accuracy on the test data. The model is pretty good at recognizing digits in unseen datasets.