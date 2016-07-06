---
layout: page
title: Look up Table
subtitle: Machine Learning Cheatsheet
---

## LDA

In natural language processing, latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar.

## LSA

Latent semantic analysis (LSA) is a technique in natural language processing, in particular distributional semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms.

## Perplexity

Perplexity is a measure of how well the model has fit the data: a model with lower perplexity assigns higher likelihood to the test responses, so we expect it to be better at predicting responses. Intuitively, a perplexity equal to *k* means that when the model predicts the next word, there are, on average *k* likely candidates. In particular, for the ideal scenario of perplexity equal to 1, we always know exactly what should be the next word. 

* [Smart Reply : Automated Response Suggestion for Email](https://arxiv.org/abs/1606.04870)

## Beam Search

In an RNN decoder, we eat up some input sequence and predict the output sequence word-by-word, conditioning on each word we predict. However, this does not necessarily give the final sequence with the best probability according to the model (since the most likely overall sentence might not begin with the single most likely word).

However, enumerating and scoring the entire set of output sentences is exponential in the length of the sequence, and we can't use dynamic programming to solve it since the nonlinearities in the RNN remove all the conditional independences (since we have unobserved hidden state).

So, beam search is a form of greedy search that does not give an exact highest probability output sequence, but lets us get some number of candidates *b*, called the beam size. What we do is instead of computing the most likely first word, we compute the *b* most likely first words (this set of *b* most likely candidates is called the **beam**). Then to compute the second word, we condition in turn on each of those *b* first words, and score all the sequences of length 2 obtained that way (b\*n of them if n is the size of the vocabulary), then we take the top *b* of those. Now we can compute the b\*n highest scoring length-3 sequences whose length-2 prefixes are in our previous beam, and take the highest-scoring *b* of those, etc.

Repeating this process takes only b times as much computation as greedy decoding, and gives us a list of *b* candidate output sequences at the end that are all guaranteed to be at least as good as the greedy decode. In practice this can help for tasks when the output has fairly complicated structure (including classifier-based parsing with non-RNN based models).

* [Beam Search @reddit](https://www.reddit.com/r/MachineLearning/comments/3a4x8t/beam_search/)
