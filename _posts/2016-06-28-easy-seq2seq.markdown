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
- [ ] Seq2Seq 
	- [ ] Description, Figure, Reference, Implementation (bucketing, padding)
- [ ] Sampled Softmax
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

Rule based models make it easy for anyone to create a bot. But it is incredibly difficult to create a bot that answers complex queries. The pattern matching is kind of weak and hence, AIML based bots suffer when they encounter a sentence that doesn't contain any known patterns. Also, it is time consuming and takes a lot of effor to write the rules manually. What if we can build a bot that learns from existing conversations (between humans). This is where *Machine Learning* comes in. 

Let us call these models that automatically learn from data, **Intelligent models**. The Intelligent models can be further classified into:

1. **Retrieval-based** models
2. **Generative** models


The Retrieval-based models pick a response from a collection of responses based on the query. It does not generate any new sentences, hence we don't need to worry about grammar. The Generative models are quite intelligent. They generate a response, word by word based on the query. Due to this, the responses generated are prone to grammatical errors. These models are difficult to train, as they need to learn the proper sentence structure by themselves. However, once trained, the generative models outperform the retrieval-based models in terms of handling previously unseen queries and create an impression of talking with a human (a toddler may be) for the user.

Read this article : [Deep Learning For Chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/) by Denny Britz where he talks about the length of conversations, open vs closed domain dialogs, challenges in generative models like Context based responses, Coherent Personality, understanding the Intention of user and how to evaluate these models. 


## Seq2Seq

Sequence To Sequence model introduced in [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078) has since then, become the Go-To model for Dialogue Systems and Machine Translation. It consists of two [RNNs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)(Recurrent Neural Networks) : An Encoder and a decoder. The encoder takes a sequence(sentence) as input and processes one symbol(word) at each timestep. Its objective is to convert a sequence of symbols into a fixed size feature vector that encodes only the important information in the sequence while losing the unnecessary information. You can visualize data flow in the encoder along the time axis, as the flow of local information from one end of the sequence to another. 

![](/img/seq2seq/seq2seq1.png)

Each hidden state influences the next hidden state and the final hidden state can be seen as the summary of the sequence. This state is called the context or thought vector, as it represents the intention of the sequence. From the context, the decoder generates another sequence, one symbol(word) at a time. Here, at each time step, the decoder is influenced by the context and the previously generated symbols.

![](/img/seq2seq/seq2seq2.png)










