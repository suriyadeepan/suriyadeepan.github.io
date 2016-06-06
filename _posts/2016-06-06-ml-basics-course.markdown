---
layout: post
title: Get Started with Machine Learning
subtitle: An introductory course on Machine Learning
tags: ["machine learning", "course 1"]
published: true
---

## Syllabus

1. [Linear Algebra basics](#linalg)
1. Linear Regression
2. Cost function
3. Gradient Descent
4. Multivariate Linear Regression
5. Logistic Regression
6. Neural Network 
7. Backpropagation

## Linear Algebra Revision<a name="linalg"></a>

To study advanced machine learning, it is recommended to know a few concepts in Linear Algebra like Matrix calculus, Eigen Decomposition, etc. But for this course, we just need to remember a few matrix operations like matrix addition, multiplication, inversion from high school. Let's start with how we represent a matrix. A matrix is typically of 2 dimensions, represented by [m,n]. A matrix with just one row or one column is called a vector. 

To practice these operations, instead of pen and paper approach, let us run the operations using [numpy](http://www.numpy.org/). Numpy is a python library that lets us create and operate on large matrices efficiently. Open up a terminal to get started.

{% highlight bash %}
# remove numpy 
sudo apt-get remove python-numpy
# install latest version through pip
sudo pip install numpy -U
# open ipython 
ipython
{% endhighlight %}

{% highlight python %}
import numpy as np
np.zeros([3,3])
# should give zero matrix of shape [3x3]
np.eye(4)
# should give a [4x4] identity matrix
np.random.random([2,3])
# should give a [2x3] matrix of random numbers
{% endhighlight %}

### Addition

$$
A + I = C\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} + \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} = \begin{bmatrix}2 & 2 & 3\\4 & 6 & 6 \\ 7 & 8 & 10\end{bmatrix}
$$

{% highlight python %}
# Addition in numpy
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
i = np.eye(3)
a+i
{% endhighlight %}

### Subtraction

$$
A - I = D\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} - \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} = \begin{bmatrix}0 & 2 & 3\\4 & 4 & 6 \\ 7 & 8 & 8\end{bmatrix}
$$

{% highlight python %}
# Subtraction in numpy
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
i = np.eye(3)
a-i
{% endhighlight %}

### Multiplication

$$
A \cdot I = I \cdot A = A\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} \cdot \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} = \begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}\\\\
A \cdot B = E\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} \cdot \begin{bmatrix}6 & 5 & 7\\2 & 8 & 8 \\ 0 & 3 & 9\end{bmatrix} = \begin{bmatrix}6 & 10 & 21\\8 & 40 & 48 \\ 0 & 24 & 81\end{bmatrix}\\
$$

{% highlight python %}
# Muliplication with identity
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
i = np.eye(3)
print a*i
print i*a
# Multiplication a.b
b = np.array([[6, 5, 7],[2, 8, 8],[0, 3, 9]])
print np.dot(a,b)
{% endhighlight %}

### More Operations

{% highlight python %}
# Transpose of matrix
x = np.arange(1,10).reshape([3,3])
print x
print x.T
# Inverse of a matrix
y = np.linalg.inv(x)
print y
{% endhighlight %}
<br>

## Linear Regression

Regression is a statistical method for modeling the relationship between a set of (independent) variables. Linear regression basically assumes a linear model. What is linear model? A model that is linear in parameters but can still use the independent variables in non-linear forms like $$x^{2}$$ and $$\log x$$. 

The model is defined by _Hypothesis_, which is a function of variables and parameters. 
$$
H = a + bx 
$$

In the equation above, [a,b] is the set of parameters and x is the independent variable. Our objective in regression, is to find the parameters [a,b]. Usually we are given a table of values of variables(x) and outputs(y) and asked to build a model that approximates the relationship between x and y. 

Download the whole table from [here](). 

| x | y | 
| :--------- |:--------- |
|6.1101|17.592|
|5.5277|9.1302|
|8.5186|13.662|
|7.0032|11.854|
|5.8598|6.8233|
|....|....|
|5.8707|7.2029|
|5.3054|1.9869|
|8.2934|0.14454|
|13.394|9.0551|

First we define a hypothesis, $$h=a + bx$$. Now our objective is to find the best values of a and b that closely fits the relationship between x and y, in other words find a,b such that $$h \approx y$$.

## Cost Function

The _cost_ or _loss_ or _error_ function calculates the difference between the hypothesis and the actual model (i.e) How wrong are the values of a and b? For linear regression, we define cost function as:

$$
L = (1/N)\sum_{i=1}^{N}(h-y)^2
$$

By measuring the error in the hypothesis we can adjust the parameters a and b to decrease the error. Now this becomes the core function of regression. We adjust the parameters, check the error, adjust them again and on and on; eventually we get the best set of parameters and hence the best fitting model. We call this iterative process, _learning_. But how exactly do we adjust the parameters?

## Gradient Descent

Gradient Descent is an optimization technique that improves the parameters of the model, step by step. In each iteration, a small step is taken in the direction of the local minima of the cost function. The distance of movement in each step, is called the learning rate. If the learning rate is too small, it takes a long time for the model to converge (to fit the data well) and if it is too big, the model might not converge. The value of learning rate($$\alpha$$) is thus, crucial to the learning process. 

$$
a : a - \alpha \nabla_{a}\\
b : b - \alpha \nabla_{b}
$$








