---
layout: page
title: Open-domain Generative Dialogue System
subtitle: Outline of Implementation of VHRED and Improvements
---

I have been fascinated with the [sequence to sequence](https://arxiv.org/abs/1406.1078) models and their applications. Despite the architectural limitations, seq2seq performs surprisingly well in sequence to sequence mapping tasks like Machine Translation. Perhaps the most fascinating application of seq2seq, is Generative Conversational Modeling. Introduced in [A Neural Conversational Model](https://arxiv.org/abs/1506.05869), conversational models or virtual agents or chatbots, which are strictly data-driven, have produced some interesting responses.

**Human:** *what is the purpose of life ?*

**Machine:** *to serve the greater good .* 

**Human:** *what is the purpose of living ?*

**Machine:** *to live forever*


I've been experimenting with open-domain generative chatbots, ever since I read that [paper](https://arxiv.org/abs/1506.05869). In the blog [*Chatbots with Seq2seq*](http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/), I have provided an introduction to traditional rule-based chatbots, classification of chatbots into data-driven and rule-based systems and the distinction between *Generative* and *Discriminative*/*retrieval-based* systems. I have introduced the sequence to sequence architecture, it's limitation, challenges and solutions. 

My priliminary attempt to build a generative chatbot by repurposing tensorflow's [Neural Machine Translation](https://www.tensorflow.org/tutorials/seq2seq) code, can be found [here](https://github.com/suriyadeepan/easy_seq2seq). I made use of Cornell Moview Dialogue Corpus for training. The model was difficult to train and did not converge most of the time. It provided a few meaningful responses, when it did converge.

![](http://i.imgur.com/6jRMYYl.gif)


In my [second attempt](https://github.com/suriyadeepan/practical_seq2seq), I've made use of tensorflow's seq2seq module, to create a more cleaner and simpler approach to this problem. I have documented every step of this process in this blog, [Practical seq2seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/). This approach yielded more interesting and meaningful responses. The model was trained on twitter chat corpus and open subtitles corpus. The results are available in the repository.


In February, I started exploring more advanced models built on top of seq2seq. My survey on augmented seq2seq models is available [here](http://suriyadeepan.github.io/pages/proposals/vhred/survey.pdf). In an attempt to build these models, I've learned to use tensorflow low level abstraction `raw_rnn`. This [repository](https://github.com/suriyadeepan/augmented_seq2seq) that contains implementations of some of these newer architectures.


Based on my survey, the most sophisticated architecture suitable for building generative dialog systems/chatbots is [HRED](https://arxiv.org/abs/1507.04808), Hierarchical Recurrent Encoder Decoder. HRED transcends the limits of vanilla seq2seq, by modeling the dialogue process, as a sequence of utterances, where an utterance is a collection of words. This approach, leads to more meaninful and consistent conversations, instead of one-off mapping. My survey mentioned above, explains the technical details of HRED.


Recently, I've come across a new architecture named VHRED, introduced in [*A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues*](https://arxiv.org/abs/1605.06069), a variant of HRED, has claimed to perform better than HRED, when evaluated by a group of human evaluators.


## Model

Chatbots or Dialog Systems, can be broadly classified into two types : Generative and Discriminative. The discriminative system learns to pick an appropriate response, to a query, from a long list of predefined responses. A generative system on the other hand, generates a response word by word, conditioned on the query. This opens up a possibility of realistic and flexible interactions. The HRED architecture, is basically modeled as a cognitive system, by it's creators. It is expected to carry out the tasks of Natural Language Understanding, Reasoning, Decision Making and Natural Language Generation. These are the components of a typical cognitive architecture. 

In spite of the sophisticated nature of HRED, the authors of VHRED remark that, it still follows the process of Shallow Generation, similar to the vanilla seq2seq model. The shallow generation, refers to the sequential word by word generation process, performed by the decoder of HRED. A dialogue, as mentioned above, is a sequence of sub-sequences, where the sub-sequences are utterances, consisting of words. The structure of an utterance, is characterized by the local statistics of language, like word co-occurrences, while the structure of the dialogue, depends on higher level concepts like, conversation topic and motivation. In HRED and vanilla seq2seq, this variation is injected only on the word level, hence, the topic or the speaker motivation, is decided incrementally while the words are being generating in a response. This leads to topic incoherence and inconsistent speaker goals.

![](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/36818eaf6376aeeaffed2523d28bebae7c9db8d7/2-Figure1-1.png)

VHRED attempts to solve this problem, by introducing an intermediate step between the context RNN and the decoding process. A continuous high-dimensional stochastic latent variable is sampled during each utterance. The decoder is conditioned on this latent variable while generating a response. The authors believe that this could lead to meaningful, long and diverse responses. The latent variable should ideally represent the high level semantic content of the response. Additionally, this process helps in maintaining the dialogue state and also alleviates the decoder state bottleneck, ie. the decoder state is responsible for summarising the past, in order to generate the next words (short term objective) and also, to sustains an optimal output trajectory, that facilitates generation of future tokens. 


## Dataset

A authors of VHRED, have made the [Twitter Dialogue Corpus](http://www.iulianserban.com/Files/TwitterDialogueCorpus.zip) and preprocessed [Ubuntu Dialog Corpus](http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip), available for public use. The MovieTriples dataset is also available for research purposes only upon request, but the authors have strongly recommended readers to benchmark the model, on Twitter and Ubuntu datasets, as they are substantially larger. Apart from these, 1.7 billion [reddit comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/)  and 2.8 million [OpenSubtitles](https://www.nr.no/~plison/pdfs/cl/slt2016.pdf) dataset, are available for research.



## References

1. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
2. [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
3. [Blog : Chatbots with Seq2seq](http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)
4. [Tensorflow's seq2seq tutorial](https://www.tensorflow.org/tutorials/seq2seq)
5. [easy_seq2seq code](https://github.com/suriyadeepan/easy_seq2seq)
6. [practical_seq2seq code](https://github.com/suriyadeepan/practical_seq2seq)
7. [Blog : Practical seq2seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/)
8. [Survey on Augmented Sequence to Sequence Models for Generative Dialog Systems](http://suriyadeepan.github.io/pages/proposals/vhred/survey.pdf)
9. [augmented_seq2seq code](https://github.com/suriyadeepan/augmented_seq2seq)
10. [Hierarchical Recurrent Encoder Decoder](https://arxiv.org/abs/1507.04808)
11. [A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](https://arxiv.org/abs/1605.06069)
12. [Twitter Dialogue Corpus](http://www.iulianserban.com/Files/TwitterDialogueCorpus.zip)
13. [Ubuntu Dialog Corpus](http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip)
14. [Every publicly available reddit comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/)
15. [Automatic Turn Segmentation for Movie & TV subtitles](https://www.nr.no/~plison/pdfs/cl/slt2016.pdf)
