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

# zero matrix
zero_mat = np.zeros([4,3])
print zero_mat

# one matrix
ones_mat = np.ones([3,5]) 
print ones_mat

# constant matrix
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

# continuous integers
nums = np.arange(2,18)
nums_mat = nums.reshape([4,4])
print nums_mat
# 0 to 9 reshaped to 3x3 matrix
print np.arange(9).reshape([3,3])

{% endhighlight %}


### Array Indexing

{% highlight python %}

# lets create a 4x4 matrix
x = np.arange(16).reshape(4,4)
print x

# To get the element 4 at position (2,2); 
print x[1,1] # index starts from 0

# To get the element 10 at position (3,3); 
print x[2,2] 

# To get row 2
print x[1]

# To get column 3
print x[:,2]

# To get elements 6,10,14
print x[1:,2]

# To get elements 13,14,15
print x[-1,1:]

# Boolean indexing / Conditional
print x>7
print x%2 == 0
print x[x<7]
print x[x%2 != 0]

{% endhighlight %}


### Datatypes

{% highlight python %}

# default type np.int64
x = np.array([ [1,2], [3,4] ])
print x, x.dtype

# type set to np.int32
x1 = np.array([ [1,2], [3,4] ], dtype=np.int32)
print x1, x1.dtype

# default type np.float64
y = np.array([ [1.,2], [3,4] ])
print y, y.dtype

# set to np.float32
y1 = y.astype(np.float32)
print y1, y1.dtype

# the boolean type
z = x>2
print x, x.dtype

{% endhighlight %}


### Arithmetics

{% highlight python %}

# elementwise operations
x = np.full([3,3],3)
y = np.eye(3)
z = np.ones(3)

# Add
print x + y
# Subtract
print x-y
# Multiply
print x * (x+z)
# Divide
print (x + 1) / (z + 1)

# Dot product
a = np.arange(9).reshape(3,3)
b = np.arange(12).reshape(3,4)
print np.dot(a,b) # or a.dot(b)
# try b.dot(a)

# transpose for dot product
x = np.random.randint(0,9,[3,3])
y = np.random.randint(0,9,[1,3])
# x.dot(y) wont work as x.shape[1] != y.shape[0]
print x.dot(y.T) # transpose y to shape[3,1]
# gives an output of shape [3,1]

{% endhighlight %}


### Statistics

{% highlight python %}

a = np.arange(9)
# sum
print a.sum()
# mean
print a.mean()
# stddev, variance
print a.std(), a.var()
# min and max
print a.min(), a.max()
# id of min and max elements
print a.argmin(), a.argmax()

b = np.arange(9).reshape([3,3])
# row-wise sum
b.sum(axis=1)
# column-wise sum
b.sum(axis=0)

{% endhighlight %}


### Broadcating

{% highlight python %}

x = np.arange(16).reshape([4,4])
# array[1] is broadcasted to shape [4,4] of x
print x + 1
# consider 2 arrays of shapes : [4,4] and [4,]
a = np.random.randint(0,10,[4,4])
b = np.arange(4)
print a*b
# here b is broadcasted to all the rows of a

{% endhighlight %}


### Array Masking

{% highlight python %}

'''
Array masking is the name of a special method of selection available in Python by means of a boolean mask, it allows to extract data out of an array based on certain condition.
'''
arr = np.arange(9).reshape([3,3])
print arr
# create a random boolean array
mask1 = np.random.choice([True,False],[3,3]).astype(np.bool)
print mask1
# mask over array
print arr[mask1]

# using conditions to create mask
div_4_mask = (arr%4 == 0)
print div_4_mask
print arr[div_4_mask]

# Efficient masking using np.putmask() function
#		using a <divisible by 3> mask
np.putmask(arr,arr%3 == 0,0)
print arr
# there is an option to operate on the elements that satisfy
#		the condition <divisible by 3>
#			enter the operation as parameter 3; say multiply by 10
arr = np.arange(9).reshape([3,3])
print arr
# if divisible by 3 multiply by 10
np.putmask(arr, arr%3 == 0, arr*10) 
print arr

{% endhighlight %}



## Reference

1. [Indexing Numpy Arrays](https://scipy.github.io/old-wiki/pages/Cookbook/Indexing)
2. [List of Mathematics Functions in Numpy](http://docs.scipy.org/doc/numpy/reference/routines.math.html) 
3. [Broadcasting Arrays in Numpy](http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/)
4. [Array Masking](https://www.getdatajoy.com/learn/Array_Masking)
