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

Regression is a statistical method for modeling the relationship between a set of (independent) variables. Linear regression basically assumes a linear model. What is linear model? A model that is linear in parameters but can still use the independent variables in non-linear forms like $$x^{2}$$ and $$\log x$$. 

The model is defined by _Hypothesis_, which is a function of variables and parameters. 
$$
H = a + bx 
$$

In the equation above, [a,b] is the set of parameters and x is the independent variable. Our objective in regression, is to find the parameters [a,b]. 

| x | y | 
| :--------- |:--------- |
|6.1101|17.592|
|5.5277|9.1302|
|8.5186|13.662|
|7.0032|11.854|
|5.8598|6.8233|
|8.3829|11.886|
|7.4764|4.3483|
|8.5781|12|
|6.4862|6.5987|
|5.0546|3.8166|
|5.7107|3.2522|
|14.164|15.505|
|5.734|3.1551|
|8.4084|7.2258|
|5.6407|0.71618|
|5.3794|3.5129|
|6.3654|5.3048|
|5.1301|0.56077|
|6.4296|3.6518|
|7.0708|5.3893|
|6.1891|3.1386|
|20.27|21.767|
|5.4901|4.263|
|6.3261|5.1875|
|5.5649|3.0825|
|18.945|22.638|
|12.828|13.501|
|10.957|7.0467|
|13.176|14.692|
|22.203|24.147|
|5.2524|-1.22|
|6.5894|5.9966|
|9.2482|12.134|
|5.8918|1.8495|
|8.2111|6.5426|
|7.9334|4.5623|
|8.0959|4.1164|
|5.6063|3.3928|
|12.836|10.117|
|6.3534|5.4974|
|5.4069|0.55657|
|6.8825|3.9115|
|11.708|5.3854|
|5.7737|2.4406|
|7.8247|6.7318|
|7.0931|1.0463|
|5.0702|5.1337|
|5.8014|1.844|
|11.7|8.0043|
|5.5416|1.0179|
|7.5402|6.7504|
|5.3077|1.8396|
|7.4239|4.2885|
|7.6031|4.9981|
|6.3328|1.4233|
|6.3589|-1.4211|
|6.2742|2.4756|
|5.6397|4.6042|
|9.3102|3.9624|
|9.4536|5.4141|
|8.8254|5.1694|
|5.1793|-0.74279|
|21.279|17.929|
|14.908|12.054|
|18.959|17.054|
|7.2182|4.8852|
|8.2951|5.7442|
|10.236|7.7754|
|5.4994|1.0173|
|20.341|20.992|
|10.136|6.6799|
|7.3345|4.0259|
|6.0062|1.2784|
|7.2259|3.3411|
|5.0269|-2.6807|
|6.5479|0.29678|
|7.5386|3.8845|
|5.0365|5.7014|
|10.274|6.7526|
|5.1077|2.0576|
|5.7292|0.47953|
|5.1884|0.20421|
|6.3557|0.67861|
|9.7687|7.5435|
|6.5159|5.3436|
|8.5172|4.2415|
|9.1802|6.7981|
|6.002|0.92695|
|5.5204|0.152|
|5.0594|2.8214|
|5.7077|1.8451|
|7.6366|4.2959|
|5.8707|7.2029|
|5.3054|1.9869|
|8.2934|0.14454|
|13.394|9.0551|

 






