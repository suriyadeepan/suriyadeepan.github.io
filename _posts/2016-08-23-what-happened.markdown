---
layout: post
title: "What happened?"
subtitle: "Review of the last 2 months"
tags: ["life", "activities", "camp"]
published: true
---

It has been 55 days since my last post. A lot has happened. I handled python sessions at this year's summer camps, held at Villupuram and Chennai, conducted by Free Software Foundation, Tamilnadu. As a result, my python skills have improved considerably. It is a shame that we were unable to conduct a camp at my home town, Puducherry. But there is still hope. I digress. I tried to summarize the events of last 2 months in this post.

## Web scraping

The pretty HTML5 documents you see on the internet consist of structured data. If you "inspect" the elements in a page, you can see the tags used to structure the document, are named sensibly, in a hierarchy. We can use this to our advantage and grab what we want from a page programmatically. This is what we call **Web Scraping**. I have always been fond of web scraping. I just lacked the skill. Last month I started experimenting with BeautifulSoup, a HTML parser written in python. I had a lot of fun with it, but eventually I got tired of writing logic for following links from page to page. Then I moved on to scrapy, a powerful framework for extracting the data you need from websites. Scrapy has templates for **spiders**, that crawl the websites and extract useful data. It has [item](http://doc.scrapy.org/en/latest/topics/items.html) templates, [item pipelines](http://doc.scrapy.org/en/latest/topics/item-pipeline.html#topics-item-pipeline), a [shell](http://doc.scrapy.org/en/latest/topics/shell.html) to experiment with, logging mechanisms and more. I have just scratched the surface of scrapy, but still I am able to do things that I thought were impossible. Like this:

![](https://raw.githubusercontent.com/suriyadeepan/wiki-graph/master/lowres.jpg)

*What did I just see?* More on that later.


## RF planning tool for Mesh Networks

This is a relatively recent development. We are building an RF planning tool, a free alternative to [CloudRF](https://cloudrf.com/) (cloudrf is partly open). Check out this [video demonstration](https://www.youtube.com/watch?v=i-pcIFWOkpQ) by CloudRF, to understand what I mean by RF planning. Our code is hosted here : [pymeshnet/FreeRF](https://gitlab.com/pymeshnet/FreeRF/). The infographics created by Ganesh, will speak louder than my words.

![](https://crabgrass.riseup.net/assets/322193/RF+Planning+%26+Link+Budgeting+3.png)
![](https://crabgrass.riseup.net/assets/322195/RF+Planning+%26+Link+Budgeting+6.png)


## Python Sessions

Python is a beautiful language. It lets you express yourself, without having to worry about trivialities, like memory overflow, etc. The code is concise and readable. Learning python by practicing is a rewarding experience. Instant gratification. It does have its faults.

![](/img/python/multithreading.jpg)

But the pros outweigh the cons. 

I taught the fundamentals of python to college kids at [Villupuram camp](https://fsftn.org/events/sc/2016/vpm/) and [Chennai camp](https://fsftn.org/events/sc/2016/chn/), organized by Free Software Foundation, Tamilnadu. Teaching, like learning, is also rewarding. I got to learn some interesting python constructs, that I missed, like generators, regular expressions, the infamous map-reduce-filter tools, zip, lambda, custom key sort, the multiprocessing module and more. I will write a separate blog post explaining these constructs. The exercises are available in this repository : [suriyadeepan/python3](https://github.com/suriyadeepan/python3) and the slides are available [here](https://raw.githubusercontent.com/FSFTN/Annual-Camp-2K16-Presentations/master/python3/Python3.pdf) for download. 


## Pandas

Last week I started learning pandas. Pandas is a python libray that provides easy-to-use, high level data structures and operations, for data analysis. I am following [sentdex's](https://www.youtube.com/user/sentdex/featured) video tutorial series : [Data Analysis with Python and Pandas](https://www.youtube.com/playlist?list=PLQVvvaa0QuDc-3szzjeP6N6b0aDrrKyL-). It is the best pandas tutorial out there, as far as I know. I have completed these sections:

1. Pandas Basics
2. IO basics
3. Building Dataset
4. Concatenating and Appending Dataframes
5. Joining and Merging Dataframes

Find my Jupyter notebok here : [suriyadeepan/pandas](https://github.com/suriyadeepan/pandas/blob/master/notebooks/Data%20Analysis%20with%20Python%20and%20Pandas%20Tutorial.ipynb)


## Painless Recurrent Neural Network  

No. It isn't another variant of Recurrent Neural Network. I just learned to build an RNN in a painless fashion. I found this amazing place, where deep learning enthusiasts share their models, datasets, papers and more : [tensorhub](https://tensorhub.com/). [aymericdamien](https://github.com/aymericdamien) uses LSTM to classify MNIST digits, in this [tutorial](https://tensorhub.com/aymericdamien/tensorflow-rnn). It is simple and straightforward. It does not require you to manually construct the gates in LSTM. I tried it and here is the [result](https://github.com/suriyadeepan/TF/blob/master/RNN/TensorFlow%20Example%20-%20Recurrent%20Neural%20Network.ipynb).


## AutoReveal

I hate working on large files. Like this one. Yeah, the [source](https://raw.githubusercontent.com/suriyadeepan/suriyadeepan.github.io/master/_posts/2016-08-23-what-happened.markdown) of this blog post is a large (debatable) markdown file. I like my code to be modular. When I was working on the python slides with reveal.js, I wrote a markdown file for each section. Then I had to integrate them into a single .html file. I decided to write a script that'll grab the contents of all the markdown files and put them in the .html file intelligently, based on a template. And here it is, [autoreveal.py](https://github.com/suriyadeepan/python3/blob/master/slides/autoreveal.py). Check out the settings file, [python3.ini](https://github.com/suriyadeepan/python3/blob/master/slides/python3.ini). May be I am a masochist. I did use arch once. 


## FSHM's "Untitled" event

This is the most recent event to happen. Last weekend, we conducted a small scale event to invite new students into FSHM, Free Software Hardware Movement, Puducherry. In spite of the last minute planning, the event went better than expected. We invited members of Chennai Meshnet Community to join us, for the afternoon session on Mesh networks.

![](/img/event/pymeet00.png)

Selva explained the basics of networking and the difference between traditional routing protocols and that of P2P networks. Ganesh took over and talked about spectrum, antennas and most importantly, the strategy for moving forward with mesh networks. 

> Wireless Networking enables a individual to empower and also empowers a community or local social group.

He went on to talk about what we lack as an organization and how individual contribution adds up to a collective effort. He explained the [RF planning tool](https://gitlab.com/pymeshnet/FreeRF/), that we are working on, with a few *cool* infographics. You can find them [here](https://crabgrass.riseup.net/chennaimeshnetworkcommunity/beehive-for-developers#a-proposal-for-free-open-tool-for-radio-planning-l) and [here](https://crabgrass.riseup.net/chennaimeshnetworkcommunity/community-guide#how-to-ease-the-commons-to-contribute-to-the-mesh-).


I think that pretty much covers the last two months. Too much information? It is okay to be a narcissist. That is what I keep telling myself.

Feel free to drop a comment. 
