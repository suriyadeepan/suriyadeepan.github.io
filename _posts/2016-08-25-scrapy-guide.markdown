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

1. [**Spider**](http://doc.scrapy.org/en/latest/topics/spiders.html) : includes the logic for crawling (following links) and scraping
2. [**Item**](http://doc.scrapy.org/en/latest/topics/items.html) : a container for the scraped data
3. [**Item Pipeline**](http://doc.scrapy.org/en/latest/topics/item-pipeline.html) : sequence of processing steps that the item objects will go through before being saved to disk
4. [**Selectors**](http://doc.scrapy.org/en/latest/topics/selectors.html) : extract data from HTML content; I use Beautiful Soup as my selector
5. [**Shell**](http://doc.scrapy.org/en/latest/topics/shell.html) : interactive shell for trying out your scraping code

Apart from these, there are a lot more features in scrapy. Find the list of all the features and services provided by scrapy, [here](http://doc.scrapy.org/en/latest/#basic-concepts).


Lets learn the workflow of scraping a website, by completing a few trivial exercises.

## Exercise set 1

In this set of exercises, we will identify the elements in a wiki page that we need, and then try to grab it from an *ipython* console. 

### Get the table of contents from this [wiki page](https://en.wikipedia.org/wiki/Transhumanism)

Open up the page, right click anywhere inside table of contents and inspect. It will open up chrome/firefox dev tools. Notice the hierarchy of HTML tags. Now we know which element to fetch. The *div* of class *toc* contains the table of contents. Lets grab it using BeautifulSoup.

![](/img/scrapy/wiki_img1.png)

{% gist fa31820275e02b0c5d1ba301bb484fdc %}


### Get all the images

Inspect an image element. You will find the 'img' tag behind it, which contains the image url in its 'src' attribute.

{% gist b940caf6cba552527613c1f93e26cc80 %}


### Get all the references

Inspect the items(span) in the **References** section.

{% gist 1361cf391517aeb11d271b97b15f7fc0 %}

### Get all the links to other wiki pages, with Title

This task is a bit trickier. We need to filter out all the unnecessary links. We just need the links to other wiki articles. Our search will be focused on just 'p' tags. We will choose only the 'a' tags that contain *title* and *href* attributes. 

{% gist f2c4baeccb7a0f5eb74ab4b3ab716997 %}


## Reference

1. [Scrapy Documentatin](http://doc.scrapy.org/en/latest/)
