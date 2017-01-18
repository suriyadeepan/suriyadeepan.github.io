---
layout: post
title: "The Principle of Maximum Likelihood"
subtitle: "Linear Regression : The Probabilistic Perspective"
tags: ["machine learning", "probability", "maximum likelihood", "theory", "mle"] 
published: true
---

TODO : Motivation

The principle of Maximum Likelihood is at the heart of Machine Learning. It guides us to find the best model in a search space of all models. In simple terms, Maximum Likelihood Estimation or MLE lets us choose a model (parameters) that explains the data (training set) better than all other models. For any given neural network architecture, the loss/cost/objective function is derived based on the principle of Maximum Likelihood.

MLE is a tool based on probability. There are a few concepts in probability, that should be understood before diving into MLE. Probability is a framework for meauring and managing uncertainty. In machine learning, every inference we make has some degree of uncertainty associated with it. It is essential for us to quantify this uncertainty. Let us start with the Gaussian Distribution.


## Gaussian Distribution

A probability distribution is a function that provides us the probabilities of all possible outcomes of a stochastic process. It can be thought of as a description of the stochastic process, in terms of the probabilities of events. The most common probability distribution is the Gaussian Distribution or the Normal Distribution. 

The gaussian distribution is a means to measure the uncertainty of a variable that is continuous between $$ -\infty$$and $$ +\infty $. The distribution is centered at mean, $$ \mu $. The width depends on the parameter$$\sigma $, the standard deviation (variance -$$\sigma^{2} $). Naturally, area under the curve equals 1. 

![](/img/mle/normal1.png)

$$ 
p(x) = \frac{1}{\sqrt{2\pi\sigma^{2}}}\
\int_{-\infty}^{+\infty} p(x) dx = 1\
$$

### Random Sampling

We can sample from this distribution. It can be written as,$$x \sim \mathcal{N}(\mu, \simga^{2}) $. 'x' is a value sampled or generated or simulated from the guassian distribution. As we sample from this distribution, most samples will fall around the center, near the mean, because of higher probability density in the center.

![](/img/mle/sampling1.png)


### Bivariate Gaussian

![](/img/mle/bivariate3d.png)

The 2D gaussian distribution or bivariate distribution, consists of 2 random variables x1 and x2. It can have many different shapes. The shape depends on the covariance matrix, $\Sigma$. Notice the variation in the shape of the gaussian with $\Sigma$, in the figure below. The mean of x1 and x2 ($\mu_1$ and $\mu_2$) determine the center of the gaussian, while $\Sigma$ determines the shape.

![](/img/mle/bivariate-shapes.png)


### Multivariate Gaussian

The Multivariate Gaussian Distribution is a generalization of bivariate distribution, for n dimensions. It is given by,

TODO : insert multivariate gaussian expression
```
y,$$ \mu $$,$$ \simga $$- random vector
$$ \simga$$- shape of the curve
```

In the above expression, **y** and **$$ \mu $$** are vectors of n dimensions, and $$ \Sigma $$ is a matrix of shape nxn.

## Maximum Likelihood

I am borrowing this amazing toy example from Nando de Fretais's [lecture](#), to illustrate MLE. Consider 3 data points, y1=1, y2=0.5, y3=1.5, which are independent and drawn from a guassian with unknown mean $$\theta$$ and variance 1. Let's say we have two choices for $$\theta$$ - 1, 2.5. Which would you choose? Which model ($$\theta$$) would explain the data better?

In general, any data point drawn from a gaussian with mean $$\theta$$ and and variance 1, can be written as,

$$ y_i \sim \mathcal{N}(\theta, 1) = \theta + \mathcal{N}(0,1) $$

The likelihood of data (y1,y2,y3) having been drawn from $$\mathcal{N}(\theta,1)$$, can be defined as,

$$P(y1,y2,y3 \vert \theta) = P(y1 \vert \theta) P(y1 \vert \theta) P(y1 \vert \theta) $$

as y1, y2, y3 are independent.

![](/img/mle/toy-eg.png)

From the figure, we see that the likelihood, product of probabilities of data given model, is basically the product of heights of green lines. It is obvious that the likelihood of model $$\theta = 1$$ is higher.
We choose the model ($$ \theta $$), that maximizes the likelihood.


## Linear Regression

Consider the case of Univariate Linear Regression. Let us make some assumptions. 

1. Each example in the data is of form (**x**,y), where **x** is a vector and y is a scalar
2. Each label in the dataset, $$y_i$$ is drawn from a gaussian distribution, with mean $$x_{i}^{T}\theta$$ and variance $$\sigma^{2}$$
3. The data points are all independent

$$ y_i = \mathcal{N}(x_i^{T}\theta, \sigma^{2}) = x_i^{T}\theta + \mathcal{N}(0, \sigma^{2}) $$

![](/img/mle/lr1.png)

As the data points are independent, we can write the joint probability distribution of $y, \theta, \sigma$ as,

$$
p(y \vert X, \theta, \sigma) = \prod_{i=1}^{n} p(y_i \vert x_i, \theta, \sigma)\

$$
TODO : complete this

![](/img/mle/lr2.png)

Each point $y_i$ is gaussian distributed, the process of learning is the process of maximizing the product of the green bars.

Typically in MLE, we maximize the log of maximum likelihood. 

Log Likelihood is given by,

$$
l(\theta) = 
$$
TODO : complete this

Observe that the first term does not depend on $\theta$. And the second term is a quadratic function of \theta, which can be drawn as a parabola.

The maxima can be found by equating the derivative of $\l(\theta)$ to zero. 

$$
\frac{dl(\theta)}{d\theta} = 0
\hat{\theta_{ML}} = (X^TX)^{-1}X^Ty
$$
TODO : complete this

Similarly, we can get the maximum likelihood of $\sigma$ (measure of uncertainty).

$$
\sigma^2 = frac{1}{n} 
$$
TODO : complete this

## Inference

![](/img/mle/inference.png)
