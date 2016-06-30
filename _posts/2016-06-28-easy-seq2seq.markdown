---
layout: post
title: "Chatbots with Seq2Seq"
subtitle: "Learn to build a chatbot using TensorFlow"
tags: ["numpy", "machine learning"]
published: true
---

{% comment %}

As I am writing this blog post, my GTX960 is training a 2-layered LSTM based Seq2Seq model using Cornell Movie Dialog Corpus dataset. Thanks to this [tutorial](https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html) on Sequence-to-Sequence Models using Tensorflow, I now understand **Sampled Softmax** and **Bucketing**. I borrowed 3 files from [tensorflow/tensorflow/models/rnn/translate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate) : [data_utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py), [seq2seq_model.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py) and [translate.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py). 

{% endcomment %}

- [ ] Chatbots Intro
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

{% comment %} Chatbots Intro {% endcomment %}

Last year, Telegram released its [bot API](https://core.telegram.org/bots/api), providing an easy way for developers, to create bots by interacting with a bot, the [Bot Father](https://telegram.me/botfather). Immediately people started creating abstractions in nodejs, ruby and python, for building bots. We (Free Software Community) created a group for interacting with the bots we built. I created **[Myshkin](https://github.com/suriyadeepan/myshkin)** in nodejs that answers any query with a quote. The program uses the linux utility **[fortune](https://en.wikipedia.org/wiki/Fortune_(Unix))**, a pseudorandom message generator. It was dumb. But it was fun to see people willingly interact with a program that I've created. Someone made a **Hodor bot**. You probably figured out what it does. Then I encountered another bot, [Mitsuku](http://www.mitsuku.com/) which seemed quite intelligent. It is written in **AIML** (Artificial Intelligence Markup Language); an XML based "language" that lets develpers write rules for the bots to follow. Basically, you write a PATTERN and a TEMPLATE, such that when the bot encounters that pattern in a sentence from user, it replies with one of the templates. Let us call this model of bots, **Rule based models**.














