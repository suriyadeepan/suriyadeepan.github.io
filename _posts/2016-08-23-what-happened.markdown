---
layout: post
title: "What happened?"
subtitle: "Review of the last 2 months"
tags: ["life", "activities", "camp"]
published: true
---

It has been 55 days since my last post. A lot has happened. I took python sessions at 2 summer camps, at Villupuram and Chennai, conducted by Free Software Foundation, Tamilnadu. As a result, my python skills have improved considerably. It is a shame that we were unable to conduct a camp at my home town, Puducherry. But there is still hope. 

## Web scraping

The pretty HTML5 documents you see on the internet consist of structured data. If you "inspect" the elements in a page, you can see the tags used to structure the document, are named sensibly, in a hierarchy. We can use this to our advantage and grab what we want from a page programmatically. This is what we call **Web Scraping**. I have also been fond of web scraping. I just lacked the skill. Last month I started experimenting with BeautifulSoup, a HTML parser written in python. I had a lot of fun with it, but eventually I got tired writing logic for following links from page to page. Then I moved on to scrapy, a powerful framework for extracting the data you need from websites. Scrapy has templates for **spiders**, that crawl the websites and extract useful data. It has [item](http://doc.scrapy.org/en/latest/topics/items.html) templates, [item pipelines](http://doc.scrapy.org/en/latest/topics/item-pipeline.html#topics-item-pipeline), a [shell](http://doc.scrapy.org/en/latest/topics/shell.html) to experiment with, logging mechanisms and more. I have just scratched the surface of scrapy, but still I am able to do things that I thought were impossible. Like this:

![](https://raw.githubusercontent.com/suriyadeepan/wiki-graph/master/lowres.jpg)

More on that later.


## RF planning tool for Mesh Networks

This is a relatively recent development. We are building an RF planning tool, similar to [CloudRF](https://cloudrf.com/), which is partly open. Check out this [video demonstration](https://www.youtube.com/watch?v=i-pcIFWOkpQ) by CloudRF, to understand what I mean by RF planning. Our code is hosted here : [pymeshnet/FreeRF](https://gitlab.com/pymeshnet/FreeRF/). The infographics created by Ganesh, will speak louder than my words.

![](https://crabgrass.riseup.net/assets/322193/RF+Planning+%26+Link+Budgeting+3.png)
![](https://crabgrass.riseup.net/assets/322195/RF+Planning+%26+Link+Budgeting+6.png)

