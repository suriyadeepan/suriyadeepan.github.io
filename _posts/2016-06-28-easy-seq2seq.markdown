---
layout: post
title: "Chatbots with Seq2Seq"
subtitle: "Learn to build a chatbot using TensorFlow"
tags: ["tensorflow", "machine learning", "seq2seq", "NLP"]
published: true
---

{% comment %}

As I am writing this blog post, my GTX960 is training a 2-layered LSTM based Seq2Seq model using Cornell Movie Dialog Corpus dataset. Thanks to this [tutorial](https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html) on Sequence-to-Sequence Models using Tensorflow, I now understand **Sampled Softmax** and **Bucketing**. I borrowed 3 files from [tensorflow/tensorflow/models/rnn/translate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate) : [data_utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py), [seq2seq_model.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py) and [translate.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py). 

{% endcomment %}

- [x] Chatbots Intro
- [-] Seq2Seq 
	- [x] Description
	- [x] Figure
	- [x] Padding
	- [x] Bucketing
	- [x] Word Embedding
	- [ ] Sampled Softmax
	- [ ] Reference
- [ ] Attention : Jointly Learning to Align and Translate
- [ ] Code Overview
- [ ] Bootstrapping easy_seq2seq
	- [ ] Dataset : Preprocessing
	- [ ] Configuration : ConfigParser
- [ ] Web Interface
	- [ ] Setup Flask
	- [ ] Run : Gif of results
- [ ] Reference : 1. Understanding RNN, 2. Chatbot platforms, 3. 
- [ ] Image Credits 

{% comment %} Chatbots Intro {% endcomment %}

Last year, Telegram released its [bot API](https://core.telegram.org/bots/api), providing an easy way for developers, to create bots by interacting with a bot, the [Bot Father](https://telegram.me/botfather). Immediately people started creating abstractions in nodejs, ruby and python, for building bots. We (Free Software Community) created a group for interacting with the bots we built. I created **[Myshkin](https://github.com/suriyadeepan/myshkin)** in nodejs that answers any query with a quote. The program uses the linux utility **[fortune](https://en.wikipedia.org/wiki/Fortune_(Unix))**, a pseudorandom message generator. It was dumb. But it was fun to see people willingly interact with a program that I've created. Someone made a **Hodor bot**. You probably figured out what it does. Then I encountered another bot, [Mitsuku](http://www.mitsuku.com/) which seemed quite intelligent. It is written in **AIML** (Artificial Intelligence Markup Language); an XML based "language" that lets develpers write rules for the bots to follow. Basically, you write a PATTERN and a TEMPLATE, such that when the bot encounters that pattern in a sentence from user, it replies with one of the templates. Let us call this model of bots, **Rule based model**.

Rule based models make it easy for anyone to create a bot. But it is incredibly difficult to create a bot that answers complex queries. The pattern matching is kind of weak and hence, AIML based bots suffer when they encounter a sentence that doesn't contain any known patterns. Also, it is time consuming and takes a lot of effort to write the rules manually. What if we can build a bot that learns from existing conversations (between humans). This is where *Machine Learning* comes in. 

Let us call these models that automatically learn from data, **Intelligent models**. The Intelligent models can be further classified into:

1. **Retrieval-based** models
2. **Generative** models


The Retrieval-based models pick a response from a collection of responses based on the query. It does not generate any new sentences, hence we don't need to worry about grammar. The Generative models are quite intelligent. They generate a response, word by word based on the query. Due to this, the responses generated are prone to grammatical errors. These models are difficult to train, as they need to learn the proper sentence structure by themselves. However, once trained, the generative models outperform the retrieval-based models in terms of handling previously unseen queries and create an impression of talking with a human (a toddler may be) for the user.

Read [Deep Learning For Chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/) by Denny Britz where he talks about the length of conversations, open vs closed domain dialogs, challenges in generative models like Context based responses, Coherent Personality, understanding the Intention of user and how to evaluate these models. 


## Seq2Seq

Sequence To Sequence model introduced in [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078) has since then, become the Go-To model for Dialogue Systems and Machine Translation. It consists of two [RNNs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)(Recurrent Neural Networks) : An Encoder and a decoder. The encoder takes a sequence(sentence) as input and processes one symbol(word) at each timestep. Its objective is to convert a sequence of symbols into a fixed size feature vector that encodes only the important information in the sequence while losing the unnecessary information. You can visualize data flow in the encoder along the time axis, as the flow of local information from one end of the sequence to another. 

![](/img/seq2seq/seq2seq1.png)

Each hidden state influences the next hidden state and the final hidden state can be seen as the summary of the sequence. This state is called the context or thought vector, as it represents the intention of the sequence. From the context, the decoder generates another sequence, one symbol(word) at a time. Here, at each time step, the decoder is influenced by the context and the previously generated symbols.

![](/img/seq2seq/seq2seq2.png)


There are a few challenges in using this model. The most disturbing one is that the model cannot handle variable length sequences. It is disturbing because almost all the sequence-to-sequence applications, involve variable length sequences. The next one is the vocabulary size. The decoder has to run softmax over a large vocabulary of say 20,000 words, for each word in the output. That is going to slow down the training process, even if your hardware is capable of handling it. Representation of words is of great importance. How do you represent the words in the sequence? Use of one-hot vectors means we need to deal with large sparse vectors due to large vocabulary and there is no semantic meaning to words encoded into one-hot vectors. Lets look into how we can face these challenges, one by one. 


### Padding

Before training, we work on the dataset to convert the variable length sequences into fixed length sequences, by **padding**. We use a few special symbols to fill in the sequence.

1. **EOS** : End of sentence
2. **PAD** : Filler
3. **GO**  : Start decoding
4. **UNK** : Unknown; word not in vocabulary

Consider the following query-response pair.

> **Q** : How are you? <br />
> **A** : I am fine.

Assuming that we would like our sentences (queries and responses) to be of fixed length, **10**, this pair will be converted to:

> **Q** : **[** PAD, PAD, PAD, PAD, PAD, PAD, "?", "you", "are", "How" **]** <br />
> **A** : **[** GO, "I", "am", "fine", ".", EOS, PAD, PAD, PAD, PAD **]**


### Bucketing

Introduction of padding did solve the problem of variable length sequences, but consider the case of large sentences. If the largest sentence in our dataset is of length 100, we need to encode all our sentences to be of length 100, in order to not lose any words. Now, what happens to "How are you?" ? There will be 97 PAD symbols in the encoded version of the sentence. This will overshadow the actual information in the sentence. 

Bucketing kind of solves this problem, by putting sentences into buckets of different sizes. Consider this list of buckets : **[** (5,10), (10,15), (20,25), (40,50) **]**. If the length of a query is 4 and the length of its response is 4 (as in our previous example), we put this sentence in the bucket (5,10). The query will be padded to length 5 and the response will be padded to length 10. While running the model (training or predicting), we use a different model for each bucket, compatible with the lengths of query and response. All these models, share the same parameters and hence function exactly the same way. 

If we are using the bucket (5,10), our sentences will be encoded to :

> **Q** : **[** PAD, "?", "you", "are", "How" **]** <br />
> **A** : **[** GO, "I", "am", "fine", ".", EOS, PAD, PAD, PAD, PAD **]**


### Word Embedding

Word Embedding is a technique for learning dense representation of words in a low dimensional vector space. Each word can be seen as a point in this space, represented by a fixed length vector. Semantic relations between words are captured by this technique. The **word vectors** have some interesting properties.

> paris – france + poland = warsaw. 

*The vector difference between paris and france captures the concept of capital city.*

![](/img/seq2seq/we1.png)

Word Embedding is typically done in the first layer of the network : Embedding layer, that maps a word (index to word in vocabulary) from vocabulary to a dense vector of given size. In the seq2seq model, the weights of the embedding layer are jointly trained with the other parameters of the model. Follow this [tutorial](http://sebastianruder.com/word-embeddings-1/) by Sebastian Ruder to learn about different models used for word embedding and its importance in NLP.


{% comment %}
### Sampled Softmax
{% endcomment %}



## Attention Mechanism

One of the limitations of seq2seq framework is that the entire information in the input sentence should be encoded into a fixed length vector, **context**. As the length of the sequence gets larger, we start losing considerable amount information. This is why the basic seq2seq model doesn't work well in decoding large sequences. The attention mechanism, introduced in this [paper](Neural Machine Translation by Jointly Learning to Align and Translate), allows the decoder to selectively look at the input sequence while decoding. This takes the pressure off the encoder to encode every useful information from the input. 

![](/img/seq2seq/attention1.png)

How does it work? During each timestep in the decoder, instead of using a fixed context (last hidden state of encoder), a distinct context vector $$ c_i $$ is used for generating word $$ y_i $$. This context vector $$ c_i $$ is basically the weighted sum of hidden states of the encoder. 

$$ c_i =  \sum_{j=1}^{n} \alpha_{ij}h_{j} $$

where *n* is the length of input sequence, $$ h_j $$ is the hidden state at timestep *j*.

$$ \alpha_{ij} = \exp(e_{ij}) / \sum_{k=1}^{n} \exp(e_{ik}) $$

$$ e_{ij} $$ is the alignment model which is function of decoder's previous hidden state $$ s_ij $$ and the jth hidden state of the encoder. This alignment model is parameterized as a feedforward neural network which is jointly trained with the rest of model. 



