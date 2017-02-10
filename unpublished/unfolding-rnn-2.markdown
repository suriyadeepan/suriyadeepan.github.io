---
layout: page
title: "Unfolding RNN 2"
subtitle: "Vanilla/GRU/LSTM RNNs from scratch, using Tensorflow"
---

The first article in this series focused on the general mechanism of RNN, architectures, variants and applications. The objective was to abstract away the details and illustrate the high-level concepts in RNN. Naturally, the next step is to dive into the details. In this article, we will follow a bottom-up approach, starting with the basic recurrent operation,  building up to a complete neural network which performs language modeling.

As we have seen in the previous article, the RNNs consist of states, which are updated every time step. The state, at time step 't', is essentially a summary of the information in the sequence till 't'. At each 't', information flows from the current input and the previous state, to the current state. This flow of information can be controlled. This is called the **gating** mechanism. Conceptually, a gate is a structure that selectively allows the flow of information from one point to another. In this context, we can employ multiple gates, to control information flow from the input to the current state, previous state to current state and from current state to output. Based on how gates are employed to control the information flow, we have multiple variants of RNNs. 


In this article, we will understand and implement the 3 most famous flavors of RNN - Vanilla, GRU and LSTM. The vanilla RNN is the most basic type of RNN. It has no gates, which means, information flow isn't controlled. Information that is critical to the task at hand, could be overwritten by redudant or irrelevant information. Hence, vanilla RNNs aren't used in practice. Though, they are very useful in learning RNNs. GRU and LSTMs consist of multiple gates that enable selective forgetting or remembering information from sequence. Let's start with vanilla RNN. 

## Vanilla RNN

At each time step, the current state is the sum of linear transformations of current input and the previous state, parameterized by weight matrices **U** and **W** respectively, followed by a non-linearity (hyperbolic tangent). The output at 't', is a similar linear transformation of current state $$s_t$$, followed by a softmax, which converts logits into normalized class probabilities. The class prediction at each time step, $$\hat{y_t}$$, assuming that we are dealing with a classification problem (sequence labeling), is calculated with the argmax function operates over $$o_t$$.

At time step 't',

state, $$s_t = tanh(Ux_t + Ws_{t-1})$$

ouput, $$o_t = softmax(Vs_t)$$

estimated class, $$\hat{y_t} = argmax(o_t)$$
