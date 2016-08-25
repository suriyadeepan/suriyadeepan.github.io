---
layout: post
title: "Web scraping with Scrapy and Beautiful Soup"
subtitle: "A Practical Guide to Web scraping"
tags: ["web scraping", "scrapy", "Beautiful Soup"]
published: true
---

Web scraping is a technique to extract information from web pages. There are more than 1 billion websites on the internet, that we know of. There are billion and billions of static and dynamically generated documents online. In this huge pile of HTML content, how can we gather information that is remotely useful to us? First of all, the site owners want you to access their site, for one reason or another. They annotate their documents with tags and catchy keywords, that will help search engines deliver content relevant to you. Second, there is structure to every HTML page. The web developers use a hierarchy of semantically meaningful tags to structure their content. We can make use of this structure, to extract content that are useful to us.

### Why would I do that?


I don't know why people scrape the web. I just know why I do it. I am a Machine Learning researcher. My models are data-hungry. More the data, better the model. Web scraping goes hand in hand with Machine Learning. For people working in Natural Language Processing, the Internet is a gold mine. You just need the right tools, to get it to cough up the good stuff.


This blog post will cover the workflow of scraping a website, step by step. 

1. **Reconnaissance** : After deciding the kind of information we want, We find a page where we can start. We will then inspect the elements that matter to us and find out their tag (div, p, etc) and the class if necessary. Open up a scrapy shell and try to get the information we need, by accessing the corresponding element using *xpath*. 
2. **Crawling** : Then, we use this logic in our code, to extract data recursively. Typically we will jump from page to page, by extracting links that match a pattern.
3. **Aquisition** : During this process, any useful information we need, say text, images, etc, will be downloaded and saved to disk. 


## Tools



