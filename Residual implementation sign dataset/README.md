i will write Residual network for image classification. In simple words, this network embody a simple path, and a skip connection. This helps in preventing vanshing and exploding gradients problems.

-Implement the basic building blocks of ResNets in a deep neural network using Keras
-Put together these building blocks to implement and train a state-of-the-art neural network for image classification
-Implement a skip connection in your network


In recent years, neural networks have become much deeper, with state-of-the-art networks evolving from having just a few layers (e.g., AlexNet) to over a hundred layers.
The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output). However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent prohibitively slow. More specifically, during gradient descent, as you backpropagate from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode," from gaining very large values).
During training, you might therefore see the magnitude (or norm) of the gradient for the shallower layers decrease to zero very rapidly as training proceeds.

residual implementation
First component of main path:
 -The first CONV2D has  F1  filters of shape (1,1) and a stride of (1,1). Its padding is "valid". Use 0 as the seed for the random uniform initialization: kernel_initializer = initializer(seed=0).
 -The first BatchNorm is normalizing the 'channels' axis.
 -Then apply the ReLU activation function. This has no hyperparameters.
Second component of main path:
 -The second CONV2D has  F2  filters of shape  (f,f)  and a stride of (1,1). Its padding is "same". Use 0 as the seed for the random uniform initialization: kernel_initializer = initializer(seed=0).
 -The second BatchNorm is normalizing the 'channels' axis.
 -Then apply the ReLU activation function. This has no hyperparameters.
Third component of main path:
 -The third CONV2D has  F3  filters of shape (1,1) and a stride of (1,1). Its padding is "valid". Use 0 as the seed for the random uniform initialization: kernel_initializer = initializer(seed=0).
 -The third BatchNorm is normalizing the 'channels' axis.
 -Note that there is no ReLU activation function in this component.
Final step:
 -The X_shortcut and the output from the 3rd layer X are added together.
 -The syntax will look something like Add()([var1,var2])
 -Then apply the ReLU activation function. This has no hyperparameters.
