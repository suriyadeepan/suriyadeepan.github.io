---
layout: page
title: Beam Search
---

Beam search is a heuristic search algorithm that explores the graph by expanding a limited number of nodes. It is a variant of breadth first search, [BFS](). At each level, it selects a predetermined number of best states which it expands (**beam width**). With an infinite beam, all the states are expanded and the search becomes identical to BFS. 

In [seq2seq](#) model, while inference, based on context, the next word can be predicted by choosing the highest probable word. This method is **greedy**. It may not produce optimal results as it doesn't consider how probable the whole decoded output is. Beam Search can be used here to select top **k**, words from context and a set of next words can be inferred given these **k** previous words. At the end of decoding, we can select the most probable sentence.

![](/img/graph/greedy.png)

![](/img/graph/beamsearch.png)
