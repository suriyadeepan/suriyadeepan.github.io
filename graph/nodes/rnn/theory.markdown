---
layout: page
title: Recurrent Neural Network
---

Recurrent Neural Networks (RNN) introduce a notion of time to the model. Adjacent time steps are connected together through recurrent connections. At any instant **t**, the network receives input from the current data point and the previous hidden state $$h^{(t-1)}$$. The following two equations represent the forward propagation step in an RNN.

1. Input sequence, $$x^{(1)}, x^{(2)},... x^{(T)}$$
2. Target sequence, $$y^{(1)}, y^{(2)},... y^{(T)}$$
3. Maximum length of sequence, $$T$$
4. Predicted data point, $$\hat{y}^{(t)}$$


$$
h^{(t)} = \sigma( W^{hx}x^{(t)} + W^{hh}h^{(t-1)} + b_{h} )\\
\hat{y}^{(t)} = softmax( W^{yh}h^{(t)} + b_{y} )\\
$$

The RNN *unfolds* into a deep feed forward network with one layer for each time step. The unfolded network can be trained using Backpropagation. This is known as Backpropagation through Time, [BPTT](#).

![](/img/graph/rnn_step.png)


## Training RNN

RNNs are notoriously difficult to train. Why? It is difficult to *remember* long-term dependencies because of **exploding** and **vanishing** gradients problem. While processing a sequence, during every time step, the value of $$W_{hh}$$ in figure below, remains constant. As a result, the gradients while backpropagation will either explode or vanish at some point depending on the value of \|$$W_{hh}$$\|.

![](/img/graph/rnn_gradient.png)

$$
\delta \to \infty if |W_{hh}| > 1\\
\delta \to 0 if |W_{hh}| < 0\\
$$

How do we **solve** this?

One of the techniques is to use a **Regularization** term that will keep the gradients from reaching extremes. The other technique is to use **Truncated BPTT** (TBPTT) that will stop exploding and vanishing gradients at the cost of sacrificing long-term dependencies.


### Notes

1. Saddle points vs local minima
2. Use of saddle-free Newton's method when SGD gets stuck






