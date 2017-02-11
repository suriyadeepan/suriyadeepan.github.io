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

$$
s_t = tanh(Ux_t + Ws_{t-1})\\
o_t = softmax(Vs_t)\\
\hat{y_t} = argmax(o_t)\\
$$


## Recurrence in Tensorflow

We now know what happens during each time step in a vanilla RNN. How do we implement this in tensorflow? The length of the input sequence varies for each example. The graph defined in tensorflow runs the recurrence operation over each item in the input sequence, one step at a time. The is graph should be capable of handling variable length sequences. In other words, it should dynamically unfold the recurrent operation 'T' times for each 'T' lengths of input sequences. This requires a loop in the computational graph. Tensorflow provides **scan** operation precisely for this purpose. Let us take a look at [*tf.scan*](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard9/tf.scan.md).

{% highlight python %}
tf.scan(fn, elems, initializer) # scan operation
def fn(st_1, xt): # recurrent function
    st = f(st_1, xt)
    return st
{% endhighlight %}

The usage is fairly simple. *fn* is the recurrent function that runs 'T' times. *elems* is the list of elements of length 'T'- the input sequence, over which *fn* is applied. During each iteration, the state of the system is calculated as a function of current input, *xt* and the previous state, *st_1*. This state, *st* is passed to the next recurrent operation at 't+1'. *initializer* is a parameter that sets the initial state of the system. The states and the inputs discussed here, are basically just vectors of fixed length.

## Language Modeling

Before jumping into the code, let me introduce the task of language modeling (RNNLM). The objective is capture the statistical relationship among words in a corpus, by learning to predict the next word, given a word. Based on which we will generate text one word at a time. I've used the term "word" here, but we will work with characters, simply because it is easier to work with data that way. The vocabulary will be fixed and small, and we don't have to deal with rare words. The table below contains input and output sequences from the dataset.

*TODO*

1. insert table here
2. symbol to index, vocabulary
3. intro to embedding


### Placeholders

{% highlight python %}
import tensorflow as tf
import numpy as np
xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
init_state = tf.placeholder(shape=[None, state_size],
    dtype=tf.float32, name='initial_state')
{% endhighlight %}

Placeholders are entry points through which data can be fed to the graph, during execution. The inputs xs_ and outputs ys_ are arrays of indices of symbols in vocabulary as discussed in the last section.

### Embedding

{% highlight python %}
embs = tf.get_variable('emb', [num_classes, state_size])
rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
{% endhighlight %}

The inputs are transformed into embeddings, using the embedding matrix, *embs*. *embedding_lookup* is a wrapper that basically selects a row of *embs* for each index in xs_, which is an array of indices.

### tf.scan

{% highlight python %}
states = tf.scan(step, 
        tf.transpose(rnn_inputs, [1,0,2]),
        initializer=init_state)
{% endhighlight %}

The scan function builds a loop that dynamically unfolds, to recursively apply the function *step*,  over **rnn_inputs**. The dimensions of tensor *rnn_inputs* are shuffled, to expose the sequence length dimension as the 0th dimension, to enable iteration over the elements of sequence. The tensor of form \[*batch_size, seqlen, state_size*\], is transposed to \[*seqlen, batch_size, state_size*\]. **states** returned by *scan* is an array of states from all the time steps, using which we will predict the output probabilities at each step.

### Recurrence

{% highlight python %}
xav_init = tf.contrib.layers.xavier_initializer
W = tf.get_variable('W', shape=[state_size, state_size],
     initializer=xav_init())
U = tf.get_variable('U', shape=[state_size, state_size],
     initializer=xav_init())
b = tf.get_variable('b', shape=[state_size],
     initializer=tf.constant_initializer(0.))
{% endhighlight %}

We define the weight matrices **W**, **U** and **b**, which parameterize the affine transformation given by $$Ux_t + Ws_{t-1} + b$$.

TODO : insert image that illustrates state transformation

{% highlight python %}
def step(hprev, x):
    return tf.tanh(
        tf.matmul(hprev, W) +
         tf.matmul(x,U) + b)
{% endhighlight %}

At time step 't', state given by $$s_t = tanh(Ux_t + Ws_{t-1})$$, is calculated and passed to the next step.

### Output

{% highlight python %}
V = tf.get_variable('V', shape=[state_size, num_classes], 
                    initializer=xav_init())
bo = tf.get_variable('bo', shape=[num_classes], 
                        initializer=tf.constant_initializer(0.))
states_reshaped = tf.reshape(states, [-1, state_size])
logits = tf.matmul(states_reshaped, V) + bo
predictions = tf.nn.softmax(logits)
{% endhighlight %}

The parameters of output transformation, **V** and **bo** are created. The **states** variable returned by *scan*, is of shape \[*seqlen, batch_size, state_size*\], which is squeezed into \[*seqlen\*batch_size, state_size*\], suitable for the matrix multiplication operation to follow. The reshaped states tensor is transformed into an array of logits, through matrix multiplication with *V*, given by, $$o_t = Vs_t + bo$$. The logits are transformed into class probabilities using the softmax function.

### Optimization

{% highlight python %}
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys_)
loss = tf.reduce_mean(losses)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
{% endhighlight %}

Cross entropy losses are calculated for each time step. The overall sequence loss is given by the mean of losses at each step. Note that [*sparse_softmax_cross_entropy_with_logits*](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.nn.sparse_softmax_cross_entropy_with_logits.md) function requires logits as inputs instead of probabilities.

### Training

{% highlight python %}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_loss = 0
    for i in range(epochs):
        for j in range(100):
            xs, ys = train_set.__next__()
            _, train_loss_ = sess.run([train_op, loss], feed_dict = {
                    xs_ : xs,
                    ys_ : ys.reshape([batch_size*seqlen]),
                    init_state : np.zeros([batch_size, state_size])
                })
            train_loss += train_loss_
        print('[{}] loss : {}'.format(i,train_loss/100))
        train_loss = 0
{% endhighlight %}

There isn't much to explain in the training code. We create a tensorflow session and initialize the shared variables. **train_op** does forward and backward propagation. Data is fed to *train_op* in small batches. The initial state of zeros, is explicitly fed as input to the *scan* function in the graph. 


### Checkpoint

{% highlight python %}
saver = tf.train.Saver()
saver.save(sess, ckpt_path + 'vanilla1.ckpt', global_step=i)
{% endhighlight %}

The session is saved to disk at the end of training.

### Text Generation

{% highlight python %}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    chars = [current_char]
    state = None
    batch_size = 1
    num_words = args['num_words'] if args['num_words'] else 111
    for i in range(num_words):
        if state:
            feed_dict = { 
                xs_ : np.array(current_char).reshape([1, 1]), 
                init_state : state_
                }
        else:
            feed_dict = { 
                xs_ : np.array(current_char).reshape([1,1]),
                init_state : np.zeros([batch_size, state_size])
                }
        preds, state_ = sess.run([predictions, last_state],
            feed_dict=feed_dict)
        state = True
        current_char = np.random.choice(preds.shape[-1], 1, 
            p=np.squeeze(preds))[0]
        chars.append(current_char)
{% endhighlight %}


I have ignored a few lines of code, that deals with the data. Check out the whole code at [vanilla.py](https://github.com/suriyadeepan/rnn-from-scratch/blob/master/vanilla.py).

