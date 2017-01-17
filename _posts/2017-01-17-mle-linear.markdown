---
layout: post
title: "The Principle of Maximum Likelihood"
subtitle: "Linear Regression : The Probabilistic Perspective"
tags: ["machine learning", "probability", "maximum likelihood", "theory", "mle"] 
published: false
---

TODO : Motivation

The principle of Maximum Likelihood is at the heart of Machine Learning. It guides us to find the best model in a search space of all models. In simple terms, Maximum Likelihood Estimation or MLE lets us choose a model (parameters) that explains the data (training set) better than all other models. For any given neural network architecture, the loss/cost/objective function is derived based on the principle of Maximum Likelihood.

MLE is a tool based on probability. There are few concepts in probability, that should be understood before diving into MLE. Probability is a framework for meauring and managing uncertainty. In machine learning, every inference we make has some degree of uncertainty associated with it. It is essential for us to quantify this uncertainty. Let us start with the Gaussian Distribution.


## Gaussian Distribution

A probability distribution is a function that provides us the probabilities of all possible outcomes of a stochastic process. It can be thought of as a description of the stochastic process, in terms of the probabilities of events. The most common probability distribution is the Gaussian Distribution or the Normal Distribution. 

The gaussian distribution is a way of measuring the uncertainty for a variable that is continuous between $ -\infty $ and $ +\infty $. The distribution is centered at mean, $ \mu $. The width depends on the parameter $ \sigma $, the standard deviation (variance - $ \sigma^{2} $). Area under the curve equals 1. 


TODO : univariate gaussian


$ 
p(x) = \frac{1}{\sqrt{2\pi\sigma^{2}}}
$

$ \int_{-\infty}^{+\infty} p(x) dx = 1 $

### Random Sampling

We can sample a value from this distribution. It can be written as, $ x \sim \mathcal{N}(\mu, \simga^{2}) $. 'x' is a value sampled or generated or simulated from the guassian distribution. As we sample from this distribution, most samples will fall around the center, near the mean, because of higher probability density in the center.


TODO : insert image [sampling in univariate gaussian]

### Bivariate Gaussian

TODO : insert image [bivariate gaussian 3D]

The 2D gaussian distribution or bivariate distribution, consists of 2 random variables x1 and x2. It can have many different shapes. The shape depends on the correlation between the random variables x1 and x2.

TODO : insert images [bivariate top view : correlation - {independent, positive, negative}]


## Multivariate Gaussian

The Multivariate Gaussian Distribution is a generalization of bivariate distribution, for n dimensions. It is given by,

TODO : insert multivariate gaussian expression
```
y, $ \mu $, $ \simga $ - random vector
$ \simga $ - shape of the curve
```

In the above expression, **y** and **$ \mu $** are vectors of n dimensions, and $ \sumof $ is a matrix of shape nxn.


