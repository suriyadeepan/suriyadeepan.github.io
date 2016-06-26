---
layout: post
title: "Numpy Primer"
subtitle: "Getting acquainted with Numpy"
tags: ["numpy", "machine learning"]
published: true
---

> Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. 

## Why learn Numpy?

In the [Introductory Course on Machine learning](http://suriyadeepan.github.io/2016-06-06-ml-basics-course/), we will be using numpy to build the models introduced in Andrew Ng's Machine Learning course and solve the excercises. Note that *octave* is used in that course instead. Since most of the existing machine learning frameworks today are based on python (Tensorflow, Theano, etc), I figured it would be practical to use numpy to understand these basic machine learning models. Numpy is also easy to learn and experiment with and it plays well with the Tensorflow and Theano. 

## Lets start

We will be working with arrays. Arrays are numpy objects of type **ndarray**. They contain elements of type **dtype** and have a particular **shape**. 

### Numpy arrays from python lists

{% highlight python %}
import numpy as np

# a list of numbers
x_list = [1,2,3,4,5]
print type(x_list)

# an array of numbers
x = np.array(x_list)
y = np.array([6,7,8,9])
print type(x), type(y)

print x, x.shape, x.dtype

# nested lists
nes_list = [ [1,2,3], [4,5,6], [7,8,9] ]
nes_array = np.array(nes_list)
{% endhighlight %}

### Creating Arrays


{% highlight python %}
zero_mat = np.zeros([4,3])
print zero_mat
ones_mat = np.ones([3,5]) 
print ones_mat
const_mat =  np.full([3,3], 6) 
print const_mat
# identity matrix
id_mat = np.eye(3)
print id_mat
# random matrix
rand_mat = np.random.random( [3,3] )
print rand_mat
# random integers
rand_ints = np.random.randint(1,10,[3,3])
print rand_ints
{% endhighlight %}














