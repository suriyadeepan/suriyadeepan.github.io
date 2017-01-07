---
layout: post
title: "Unfolding RNNs"
subtitle: "RNN : theory, concepts and architectures"
tags: ["machine learning", "deep learning", "rnn"] 
published: false
---

When I came to know about the mind-blowing accomplishments of deep neural networks, I was like a kid in a candy store. RNN was one of the shiniest toys, that caught my eye. I read Andrej Karpathy's blog post on the [Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), but I couldn't follow his code for text generation (Language Modeling). I was fascinated by what RNNs are capable of, and at the same time confused by how they worked. Then came the bloggers (they were always there; I just didn't notice). I came across Denny Britz's [blog](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/), from which I understood how exactly an RNN works.

This blog post is addressed to my past self that was confused about the internals of RNN. Through this blog post, I hope to help people interested in RNNs, to get an intuitive understanding of what it is, how it works, different versions of it and which version can be applied where.

**Warning** : I have organized this post based on the structure of Chapter 10 of [Ian Goodfellow's book](http://www.deeplearningbook.org/). Most concepts are directly taken from that chapter, which I have presented along with my comments. Do not be alarmed when you notice the similarities.

## Recurrence

Traditional feed-forward neural networks and variants like Convolutional Network, can be easily understood because of their static nature. A 3-layered Neural Network looks like this. We build it by stacking layer on top of another layer.

TODO : write more on how layers are stacked

$$
h1 = tanh(W_1x + b_1)
o  = tanh(W_2h1 + b_2)
\hat{y}  = softmax(o)
$$


We can express a neural network as a simple Directed Acyclic Graph(DAG), to understand the architecture and data flow. The same neural network can be expressed as a DAG below.

![](/img/rnn/ff1.png)

A recurrent neural network is a neural network with cycles in it. To understand cycles, think of a dynamic system whose present state is a function of (depends on) the previous state. It can be expressed as below:

$$ state, s_{t} = f( s_{t-1} ; \theta ) $$

Here, the state at time 't', $$s_{t}$$  is a function (f) of previous state, parameterized by $$ \theta $$. This equation can be *unfolded* as follows:

$$ s_{3} = f( s_{2}; \theta ) = f( f( s_{1}; \theta ) \theta ) $$

![](/img/rnn/unfold-1.png)

## External Signal, X

Consider a little more complex system, whose state not only depends on the previous state, but also on an external signal 'x'. 

$$ s_{t} = f( s_{t-1}, x_{t}; \theta ) $$

The system takes a signal $$ x_{t} $$ as input during time step 't', updates its state based on the influence of x_t and the previous state, s_t_1.  Now, let us try to apply what we have seen, to a neural network. The state of the system can be senn as the hidden units of a neural network. The signal x\_t at each time step, can be seen as a sequence of inputs given to the neural network, one time step at a time. At this point, we are using the term "time steps" as a synonym to steps in a sequence. 

TODO : fix the last statement

The state of such a neural network can be represented with hidden layer $$ h_{t} $$, as follows:
$$ h_{t} = f( h_{t-1}, x_{t}; \theta ) $$

TODO : motivate the next paragraph

The fixed-size hidden units used to represent the state of the network at each timestep, is essentially a lossy summary of task-relevant aspects of the past sequence of inputs. The system becomes Markovian (ie.) The next state of the system depends entirely on the present state, not the past. In other words, the current state captures everything necessary from the past. But of course, no system is truly Markovian. Hence, the state at any time step 't', is a lossy summary of the past. I feel the need to bring up how human memory works. Our memories are extremely lossy, distorted versions of the events of the past. We remember only what is necessary from the past; everything else is discarded.

Same is true here for the "memory" (internal state) of RNN. During each timestep, the internal state captures what is absolutely necessary to accomplish the task. What task? To minimize the loss. Consider the feed-forward neural network that looks at a series of images and classifies them as cats or dogs or whatever. The parameters (weights and biases) of the network are learned, to twist and turn the input images, in order to maximise the chances of correct classification.

TODO : link to Olah's blog
TODO : complete this thread of thought

## Unfolding

An RNN can be compactly represented as a cyclic circuit diagram, where the input signal is processed with a delay. This can be unfolded into a computational graph, as a series of steps. Notice the repeated appliation of function 'f'.

![](/img/unfold0.png)

Let $$ g_{t} $$ be the function that represents the unfolded graph after 't' timesteps.

Hidden state at time 't', can be written in two ways:

- as an unfolded graph
$$ h_{t} = g_{t}(x_{t}, x_{t-1}, ... x_{1}) $$

- as a recurrence relation
$$ h_{t} = f( h_{t-1}, x_{t}; \theta ) $$


## Forward Propagation

Now that we have an intuition about how RNNs function, let us move on to more concrete aspects of it. Given a sequence of inputs $$ {x_1, x_2,... x_t} $$, how do you propagate it through an RNN? How do we handle the input, update the state and produce output at each step?

![](/img/unfold.png)

What do we know? The output at each step only depends on the state at each step, as the state captures everything necessary. The state is dependent on the current input and the previous state. The state at time 't', $$ h_t $$  can be written as a function of previous state, $$ h_{t-1} $$ and current input $$ x_t $$ as follows:

$$ h_{t} = tanh ( Wh_{t-1} + Ux_{t} + b) $$
output at 't', $$ o_{t} = Vh_{t} + c $$
estimate of y, $$ \hat{y_{t}} = softmax(o_{t}) $$

The estimate, $$ \hat{y_{t}} $$ is typically a normalized probability over ouput classes. The loss compares the estimate with the ground truth (target). 

Loss, $$ L = \sum_{t} L_t = - \sum_{t} log P_{model}(y_t | {x_1,...x_t}) $$

TODO : explain the probability term; good luck with that ;) 

## Vector to sequence

So far, we have seen a typical RNN, which takes an input at each step and produces an output at each step. There is a special category of RNN that takes a fixed-length vector as input and produces a series of outputs. An application of this architecture of RNN, is the task of image captioning. Given a fixed length vector which is typically the feature vector of the input image (we can use an MLP or a CNN to extract features of an image), our RNN need to generate a proper caption, word by word. The next word depends on the previous words and also the feature vector of the image (context vector). 

![](/img/rnn/vec2seq.png)

The context, 'x' influcences the network by acting as the new "bias" parameter - (x^T)R.


## Encoder-Decoder/Seq2Seq Architecture 

There are applications where we might need to map a sequence to another sequence of different length. Recall from previous discussions that a traditional RNN emits one output for one input, at each step. Take Machine Translation for example. A sentence in English doesn't necessarily translates to a French sentence of the same length. The Encoder-Decoder model or the Sequence to Sequence (seq2seq) model, consists of two RNNs : an encoder and a decoder. The encoder processes the input sequence. The encoder doesn't emit an output at each step. Instead, it just takes the input sequence, one word at a time and tries to capture task-relevant information from the sequence, in its internal state. The final hidden state should ideally be a task-relevant summary of the input sequence. This final hidden state, is called the context or thought vector. 

The context acts as the only input to the decoder. It can be connected to the decoder in multiple ways. The initial state of the decoder can be set as a function of the context or the context can be connected to the hidden states at every time step, or both. It is important to note that the hyperparameters of the encoder and decoder can be different, depending on the type of input and output sequences. For example, we can set the number of hidden units for encoder to be different than that of the decoder. 

![](/img/rnn/seq2seq.png)


## Bidirectional RNN

The RNNs we have seen so far, have a "causal" structures, (ie.) the present is influenced by the past, but not the future. The state of the system depends on the previous inputs, state, $$ s_{t} = f(x_t, x_{t-1}, .. x_1) $$. There is special category of RNN, in which the state of the system at time 't' and therefore the output at 't', $$ o_t $$, depends on not only on inputs from the past, but also on the inputs from the future. This is confusing right? Let's forget about the past and future. Consider a input sequence, x = $$ {x_1, x_2, .., x_i,... x_n} $$.

At step i, we have 2 hidden states - $$ h_i $$ and $$ g_i $$. $$ h\_i $$ captures the flow of information from left to right $$ {x_1,x_2,...x_i} $$, while $$g_i$$ captures the information from right to left $$ {x_{i+1}, ... x_n} $$. This kind of RNN, capable of capturing information from the whole sequence, is known as a Bi-directional RNN. As the name says, it has 2 RNNs in it, for processing the sequence from either directions. At each step i, we have information from the whole sequence. Where is this applicable?

![](/img/rnn/bi.png)

In speech recognition, we may need to pick a phoneme at step i, based on inputs from i+1, i+2,... We may even have to look futher ahead, and gather information from words in the future steps, in order for the phoneme at step i, to make linguistic sense.

## Deep RNN

We need to be careful visualizing depth in an RNN. An RNN when unrolled can be seen as a deep feed-forward neural network. From this perspective, depth equals the number of timesteps. Now consider a single timestep 't'. 

![](/img/rnn/depth.png)

We can make computation at each time step, deeper by introducing more hidden layers. This type of RNN is known as Stacked RNN or more commonly, deep RNN. Typically, we have multiple layers of non-linear transformations, between hidden to hidden connections. This enables the RNN to capture higher level information (features composed of simpler features) from the input sequence, and store it as the state of the system. With great depth comes greater difficulty in training. Deeper RNNs take more time to train. But they are better in any of the tasks we have discussed, than shallow RNNs. And hence, in practice most people tend to use Stacked (multi-layer) RNNs.
