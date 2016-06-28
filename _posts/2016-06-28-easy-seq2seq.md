---
layout: post
title: "Chatbots with Seq2Seq"
subtitle: "Learn to build a chatbot using TensorFlow"
tags: ["numpy", "machine learning"]
published: true
---

As I am writing this blog post, my GTX960 is training a 2-layered LSTM based Seq2Seq model using Cornell Movie Dialog Corpus Dataset. Thanks to this [tutorial](https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html) on Sequence-to-Sequence Models using Tensorflow, I now understand **Sampled Softmax** and **Bucketing**. I borrowed 3 files from [tensorflow/tensorflow/models/rnn/translate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate) : [data_utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py), [seq2seq_model.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py) and [translate.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py). 
