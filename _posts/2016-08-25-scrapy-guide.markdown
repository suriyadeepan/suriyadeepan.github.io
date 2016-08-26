---
layout: post
title: "A Practical Guide to Web scraping"
subtitle: "Web scraping with Scrapy and Beautiful Soup"
tags: ["web scraping", "scrapy", "Beautiful Soup"]
published: true
---

Web scraping is a technique to extract information from web pages. There are more than 1 billion websites on the internet, that we know of. There are billion and billions of static and dynamically generated documents online. In this huge pile of HTML content, how can we gather information that is remotely useful to us? First of all, the site owners want you to access their site, for one reason or another. They annotate their documents with tags and catchy keywords, that will help search engines deliver content relevant to you. Second, there is structure to every HTML page. The web developers use a hierarchy of semantically meaningful tags to structure their content. We can make use of this structure, to extract content that are useful to us.

### Why would I do that?

I don't know why people scrape the web. I just know why I do it. I am a Machine Learning researcher. My models are data-hungry. More the data, better the model. Web scraping goes hand in hand with Machine Learning. For people working in Natural Language Processing, the Internet is a gold mine. You just need the right tools, to get it to cough up the good stuff.

This blog post will cover the workflow of scraping a website, step by step. 

1. **Reconnaissance** : After deciding the kind of information we want, We find a page where we can start. We will then inspect the elements that matter to us and find out their tag (div, p, etc) and the class if necessary. Open up a scrapy shell and try to get the information we need, by accessing the corresponding element using *xpath*. Alternately, we can open up a ipython console and dissect the contents of the page, with Beautiful Soup.
2. **Crawling** : Then, we use this logic in our code, to extract data recursively. Typically we will jump from page to page, by extracting links that match a pattern.
3. **Aquisition** : During this process, any useful information we need, say text, images, etc, will be downloaded and saved to disk. 

## Beautiful Soup

> Beautiful Soup is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It commonly saves programmers hours or days of work.

I started web scraping with Beautiful Soup. I tried gather content from 4 or 5 sites. I had to manually handle a lot of tasks, like rules for crawling, parallel downloads, designing a class for items that I want to extract from site, etc,. Soon I realized that Beautiful Soup wasn't enough. I needed something more powerful. A tool that will handle all the trivialities and lets me work on the logic.

## Scrapy

> Scrapy is a fast high-level web crawling and web scraping framework, used to crawl websites and extract structured data from their pages.

Scrapy is powerful and extensible. With just a few modifications, I can create a spider to crawl any website, within minutes. It has a ton of features that will make your job as a programmer, easier.

1. **Spider** : includes the logic for crawling (following links) and scraping
2. **Item** : a container for the scraped data
3. **Item Pipeline** : sequence of processing steps that the item objects will go through before being saved to disk
4. **Shell** : interactive shell for trying out your scraping code


## Reference

1. [Scrapy Documentatin](http://doc.scrapy.org/en/latest/)
