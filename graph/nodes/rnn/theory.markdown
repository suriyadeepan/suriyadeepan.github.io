---
layout: page
title: Recurrent Neural Network
---

Recurrent Neural Networks (RNN) introduce a notion of time to the model. Adjacent time steps are connected together through recurrent connections. At any instant **t**, the network receives input from the current data point and the previous hidden state $$h^{(t-1)}$$. The following two equations represent the forward propagation step in an RNN.

$$
h^{(t)} = \sigma( W^{hx}x^{(t)} + W^{hh}h^{(t-1)} + b_{h} )\\
\hat{y}^{t} = softmax( W^{yh}h^{(t)} + b_{y} )
$$

The RNN *unfolds* into a deep feed forward network with one layer for each time step. The unfolded network can be trained using Backpropagation. This is known as Backpropagation through Time, [BPTT](#).

![](/img/graph/rnn_step.png)


