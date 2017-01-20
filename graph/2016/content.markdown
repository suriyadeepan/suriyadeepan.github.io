---
layout: page
title: "2016"
---

Feed-forward Neural Networks or Multilayer Perceptrons(MLP) are the most common kind of neural networks. An MLP is capable of great many things. With increase in depth, the MLP can learn features from high dimensional data. There are 2 functions performed by the network, and hence it can be perceived as having 2 forms. Firstly, an MLP learns representation of data. It learns to capture the features in the data, from lower level details like edges, corners, small shapes, to higher level details like objects, faces, etc,. Secondly, it learns a function that maps the input to the output. Neural networks are universal function approximators. They can, in theory, learn to implement any function. Based on the noisy distribution P(x,y) represented by the training examples, the network learns to capture the original data generating distribution, without overfitting on the noise.

<iframe height='345' scrolling='no' title='first pen' src='//codepen.io/suriyadeepan/embed/vgxgMd/?height=345&theme-id=light&default-tab=result&embed-version=2' frameborder='no' allowtransparency='true' allowfullscreen='true' style='width: 100%;'>See the Pen <a href='http://codepen.io/suriyadeepan/pen/vgxgMd/'>first pen</a> by Suriyadeepan Ramamoorthy (<a href='http://codepen.io/suriyadeepan'>@suriyadeepan</a>) on <a href='http://codepen.io'>CodePen</a>.
</iframe>

The linear model of neural network, works well with the existing techniques for learning. The cost function, stays convex and hence converges to a global minimum everytime. The serious limiation of such a network, is their inability to capture non-linearity in data. To be exact, the interaction between the components of x, cannot be taken into account, by a linear model. So, no matter how many layers are stacked together, the network, as a whole remains linear. Consider a network that learns the XOR function.

TODO : write more about the inability of linear models
TODO : insert table

## Non-linearity

Non-linear functions are introduced into the network, to better describe the features of data. Typically, each layer in a neural network does an affine transformation, controlled by learned paramters, followed by an element-wise application of non-linearity. We call this the activation function (g). The most commonly used non-linearities are sigmoid, hyperbolic tangent and more recently, Rectified Linear Unit (ReLU). ReLU is the most effective non-linearity, used today in most networks. It is piecewise linear as shown in figure [fig] below. As it is nearly linear, it has all the useful properties of linear functions. For example, they are easy to optimize with gradient descent. 

$$
h = g(Wx + c)\\
g(z) = max(0,z)\\
$$

TODO : insert ReLU figure, sigmoid and tanh

## Depth

TODO : User XOR network to demonstrate how each hidden layer works together to create a space where X is linearly separable.

The primary function of the hidden layers in the network, is to work together, to create a space where X is linearly separable. The n-dimensional space in which X exists, is twisted (affine) and distorted (non-linearity), again and again, at each hidden layer, to finally create a space in which X is linear separable. 

Then, the output layer maps the output of final hidden layer to output, y. The hidden layers map the input data X, to features, h, $$ h = f(x; \theta) $$. The output layer provides additional transformation from features to the output. 


## Loss function

The most commonly used method of estimating P(y \vert x; \theta), is Maximum Likelihood. The loss function is given by the cross-entropy between training data and model's predictions.

TODO : fill in on maximum likelihood and cross-entropy loss (cce with an example)

The choice of loss function depends on the type of output units used. The commonly used output units are:

1. Linear
2. Sigmoid
3. Softmax


### Linear Unit

The simplest form of output unit, is a linear unit. It does an affine transformation without non-linearity. 







