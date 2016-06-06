---
layout: post
title: Get Started with Machine Learning
subtitle: A introductory course on Machine Learning
tags: ["machine learning", "course 1"]
published: true
---

## What are the contents?

1. Linear Algebra basics
1. Linear Regression
2. Cost function
3. Gradient Descent
4. Multivariate Linear Regression
5. Logistic Regression
6. Neural Network 
7. Backpropagation

## Linear Algebra Revision

To study advanced machine learning, it is recommended to know a few concepts in Linear Algebra like Matrix calculus, Eigen Decomposition, etc. But for this course, we just need to remember a few matrix operations like matrix addition, multiplication, inversion from high school. Let's start with how we represent a matrix. A matrix is typically of 2 dimensions, represented by [m,n]. A matrix with just one row or one column is called a vector. 

To practice these operations, instead of pen and paper approach, let us run the operations using [numpy](http://www.numpy.org/). Numpy is a python library that lets us create and operate on large matrices efficiently. Open up a terminal to get started.

```bash
# remove numpy 
sudo apt-get remove python-numpy
# install latest version through pip
sudo pip install numpy -U
# open ipython 
ipython
```

```python
import numpy as np
a = np.zeros([3,3])
b = np.eye(4)
c = np.random.random([2,3])
```

### Addition

$$
A + I = C\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} + \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} = \begin{bmatrix}2 & 2 & 3\\4 & 6 & 6 \\ 7 & 8 & 10\end{bmatrix}
$$

```python
# Addition in numpy
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
i = np.eye(3)
a+i
```

### Subtraction

$$
A - I = D\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} - \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} = \begin{bmatrix}0 & 2 & 3\\4 & 4 & 6 \\ 7 & 8 & 8\end{bmatrix}
$$

```python
# Subtraction in numpy
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
i = np.eye(3)
a-i
```

### Multiplication

$$
A \cdot I = I \cdot A = A\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} \cdot \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} = \begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}\\\\
A \cdot B = E\\
\begin{bmatrix}1 & 2 & 3\\4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} \cdot \begin{bmatrix}6 & 5 & 7\\2 & 8 & 8 \\ 0 & 3 & 9\end{bmatrix} = \begin{bmatrix}6 & 10 & 21\\8 & 40 & 48 \\ 0 & 24 & 81\end{bmatrix}\\
$$

```python
# Muliplication with identity
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
i = np.eye(3)
print a*i
print i*a
# Multiplication a.b
b = np.array([[6, 5, 7],[2, 8, 8],[0, 3, 9]])
print a*b
```


