---
layout: page
title: "The Principle of Maximum Likelihood"
subtitle: "Linear Regression : The Probabilistic Perspective"
---

> *I wrote this article a week ago and kept it as a draft till today. It felt like I've presented a series of unrelated concepts without binding them together. I am publishing it anyway, as a work in progress. I would really appreciate your help in completing this.*

We can go a long way in Machine Learning without having to deal with the scary probabilistic expressions. In order to be able to read the bleeding edge research as soon as it is published, not from a blog months later, we need to able to speak the language of probability. Probability is a great way to express the relationship between the data and the model, concisely. We are familiar with how Linear Regression works from Andrew Ng's [course](#). We know the loss function just tries to minimize the quadratic distance between data points and the model (line). In this post, we will revisit Linear Regression from a probabilistic perspective, using a method known as the Maximum Likelihood estimation. We could apply this knowledge to any Neural Network based architecture.

The principle of Maximum Likelihood is at the heart of Machine Learning. It guides us to find the best model in a search space of all models. In simple terms, Maximum Likelihood Estimation or MLE lets us choose a model (parameters) that explains the data (training set) better than all other models. For any given neural network architecture, the objective function can be derived based on the principle of Maximum Likelihood.

MLE is a tool based on probability. There are a few concepts in probability, that should be understood before diving into MLE. Probability is a framework for meauring and managing uncertainty. In machine learning, every inference we make has some degree of uncertainty associated with it. It is essential for us to quantify this uncertainty. Let us start with the Gaussian Distribution.


## Gaussian Distribution

A probability distribution is a function that provides us the probabilities of all possible outcomes of a stochastic process. It can be thought of as a description of the stochastic process, in terms of the probabilities of events. The most common probability distribution is the Gaussian Distribution or the Normal Distribution. 


<iframe height='383' scrolling='no' title='gaussian - coin toss' src='//codepen.io/suriyadeepan/embed/jymmZw/?height=383&theme-id=light&default-tab=result&embed-version=2' frameborder='no' allowtransparency='true' allowfullscreen='true' style='width: 100%;'>See the Pen <a href='http://codepen.io/suriyadeepan/pen/jymmZw/'>gaussian - coin toss</a> by Suriyadeepan Ramamoorthy (<a href='http://codepen.io/suriyadeepan'>@suriyadeepan</a>) on <a href='http://codepen.io'>CodePen</a>.
</iframe>


That was fun to watch but how is this relevant to linear regression or machine learning? The data points in the training set, do not accurately represent the original data generating distribution/process. Hence we consider the process stochastic and build our model to accomodate a certain level of uncertainty. Every data point can be considered a random variable sampled from the data generating distribution which we assume to be gaussian.


The gaussian distribution is a means to measure the uncertainty of a variable that is continuous between $$ -\infty $$ and $$ +\infty $$. The distribution is centered at mean, $$ \mu $$. The width depends on the parameter $$ \sigma $$, the standard deviation (variance, $$ \sigma^{2} $$). Naturally, area under the curve equals 1.

![](/img/mle/normal1.png)

$$ 
p(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}\\
\int_{-\infty}^{+\infty} p(x) dx = 1\\
$$


### Random Sampling

We can sample from this distribution. It can be written as, $$x \sim \mathcal{N}(\mu, \sigma^{2}) $$. 'x' is a value sampled or generated or simulated from the guassian distribution. As we sample from this distribution, most samples will fall around the center, near the mean, because of higher probability density in the center.

![](/img/mle/sampling1.png)


### Multivariate Gaussian

![](/img/mle/bivariate3d.png)

The 2D gaussian distribution or bivariate distribution, consists of 2 random variables x1 and x2. It can have many different shapes. The shape depends on the covariance matrix, $$\Sigma$$. The Multivariate Gaussian Distribution is a generalization of bivariate distribution, for 'n' dimensions.

$$
p(y) = \vert 2\pi\Sigma \vert^{-1/2}e^{-1/2(y-\mu)^T\Sigma^{-1}(y-\mu)}
$$


In the above expression, **y** and **$$ \mu $$** are vectors of 'n' dimensions, and $$ \Sigma $$ is a matrix of shape 'nxn'.


The figure below presents the top view of bivariate gaussian. The circles denote the data points sampled from the distribution. Notice the variation in the shape of the gaussian with $$\Sigma$$. The mean of x1 and x2 ($$\mu_1$$ and $$\mu_2$$) determine the center of the gaussian, while $$\Sigma$$ determines the shape.

![](/img/mle/bivariate-shapes.png)



## Maximum Likelihood

I am borrowing this amazing toy example from Nando de Fretais's [lecture](https://www.youtube.com/watch?v=voN8omBe2r4), to illustrate MLE. Consider 3 data points, y1=1, y2=0.5, y3=1.5, which are independent and drawn from a guassian with unknown mean $$\theta$$ and variance 1. Let's say we have two choices for $$\theta$$ : {1, 2.5}. Which would you choose? Which model ($$\theta$$) would explain the data better?

In general, any data point drawn from a gaussian with mean $$\theta$$ and and variance 1, can be written as,

$$ y_i \sim \mathcal{N}(\theta, 1) = \theta + \mathcal{N}(0,1) $$

The likelihood of data (y1,y2,y3) having been drawn from $$\mathcal{N}(\theta,1)$$, can be defined as,

$$P(y1,y2,y3 \vert \theta) = P(y1 \vert \theta) P(y1 \vert \theta) P(y1 \vert \theta) $$

as y1, y2, y3 are independent.

![](/img/mle/toy-eg.png)

From the figure, we see that the likelihood, product of probabilities of data given model, is basically the product of heights of dotted lines. It is obvious that the likelihood of model $$\theta = 1$$ is higher. We choose the model ($$ \theta $$), that maximizes the likelihood.


## Linear Regression

Consider the case of Univariate Linear Regression. Let us make some assumptions. 

1. Each example in the data is of form (**x**,y), where **x** is a vector and y is a scalar
2. Each label in the dataset, $$y_i$$ is drawn from a gaussian distribution, with mean $$x_{i}^{T}\theta$$ and variance $$\sigma^{2}$$
3. The data points are all independent

$$ y_i = \mathcal{N}(x_i^{T}\theta, \sigma^{2}) = x_i^{T}\theta + \mathcal{N}(0, \sigma^{2}) $$

![](/img/mle/lr1.png)

As the data points are independent, we can write the joint probability distribution of $$y, \theta, \sigma$$ as,

$$
p(y \vert X, \theta, \sigma) = \prod_{i=1}^{n} p(y_i \vert x_i, \theta, \sigma)\\
$$

TODO : complete this

![](/img/mle/lr2.png)

Each point $$y_i$$ is gaussian distributed, the process of learning is the process of maximizing the product of the green bars.

Typically in MLE, we maximize the log of maximum likelihood. 

Log Likelihood is given by,

$$
l(\theta) = 
$$
TODO : complete this

Observe that the first term does not depend on $\theta$. And the second term is a quadratic function of $$\theta$$, which can be drawn as a parabola.

The maxima can be found by equating the derivative of $$l(\theta)$$ to zero. 

$$
dl(\theta)/d\theta = 0\\
\hat{\theta_{ML}} = (X^TX)^{-1}X^Ty\\
$$
TODO : complete this

Similarly, we can get the maximum likelihood of $$\sigma$$ (measure of uncertainty).

$$
\sigma^2 = \frac{1}{n} 
$$
TODO : complete this

## Inference

![](/img/mle/inference.png)
