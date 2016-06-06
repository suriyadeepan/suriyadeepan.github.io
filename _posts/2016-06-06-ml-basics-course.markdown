---
layout: post
title: Get Started with Machine Learning
subtitle: A introductory course on Machine Learning
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

Regression is a statistical method for modeling the relationship between a set of (independent) variables. Linear regression basically assumes a linear model. What is linear model? A model that is linear in parameters but can still use the independent variables in non-linear forms like $$x^{2} and \log x$$.


