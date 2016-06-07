---
layout: post
title: "Flask : Quick Start"
subtitle: A quick intro to flask
tags: ["python", "flask", "web development"]
published: true
---

Yesterday 6th June, I participated in a tutorial session on [Flask](http://flask.pocoo.org/), via google hangout (What happened to the chat option?). Ramaseshan arranged the session; he invited [Abhinav]() who uses flask for his web applications, as the speaker. What is Flask? 

> Flask is a microframework for Python based on Werkzeug, Jinja 2 and good intentions. And before you ask: It's BSD licensed!

It is a lightweight web framework for people who think [Django](https://www.djangoproject.com/) is a way too complicated tool for the job. Most of my code in Deep learning is exclusively written in python (numpy, theano, keras, scipy). And I don't want to get too deep into the "web" side of things. Hence, Flask is an ideal tool for me.

Abhinav covered the following topics in yesterday's session.

1. Getting the Hello World Application running
2. Serving static files
3. Rendering Templates
4. Receiving URL encoded parameters
5. Basic Authentication System


I would like to add the installation procedure that I had to go through before getting started with _hello world_.

## Setting up Flask in Virtual Environment

{% highlight bash %}
# instructions for debian system
# install virtualenv 
sudo apt-get install python-virtualenv
# check version, mine is 15.0.1
virtualenv --version
# create a folder where you will store your projects
mkdir .flaskenv && cd .flaskenv
# create virtual environment "flask-env"
virtualenv flask-env
# activate virtual environment
source flask-env/bin/activate
# now we are in the virtual environment, lets install flask
pip install Flask
{% endhighlight %}


We have successfully installed flask in a virtual environment. Now lets jump to the _hello world_ applications.

## Hello World!

Create a file _helloworld.py_. Copy paste the following code into it. And run it.

{% highlight python %}
from flask import Flask

app = Flask(__name__)

@app.route("/") 
def hello(): 
    return "Hello World!"


if (__name__ == "__main__"): 
    app.run(port = 5001) 
{% endhighlight %}


{% highlight bash %}
# run it
python helloworld.py
# * Running on http://127.0.0.1:5001/ (Press CTRL+C to quit)
{% endhighlight %}





