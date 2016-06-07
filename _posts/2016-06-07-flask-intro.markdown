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

When you load _localhost:5001_ in your browser, we will see a page with "hello world" in it. 

The code is fairly simple; the imports, main function, flask app intitialization and routing. _@app.route("/")_ runs the function _hello()_ whenever a user loads the root address, that is _localhost:5001/_. 

## Serving Static Files

In this section, we will be serving a file _app.js_. Create a directory _static_, where all the static files will go. Create _app.js_ inside _static/_.

{% highlight bash %}
# create a file app.js inside directory static/
mkdir static && cd static
touch app.js
{% endhighlight %}

Copy paste the alert call into app.js. 

{% highlight javascript %}
alert("you are feeling app.js!");
{% endhighlight %}

Add _static_url_path="/static"_ parameter to _Flask()_ constructor in line 3 of _helloworld.py_. Run _helloworld.py_. Access the path _http://localhost:5001/static/app.js_ in your browser. It should display the contents of _app.js_.

{% highlight python %}
# Line 3
app = Flask(__name__,static_url_path="/static") 
{% endhighlight %}


## Rendering Templates

When we have the need to render html pages/templates, we can use the function *render_template()*. Create a folder _templates_. Create a html file _index.html_ inside it. 

{% highlight html %}
<h1>A HTML File from templates/</h1>
<script src="static/app.js"></script>
{% endhighlight %}

Notice that we have embedded our javascript file _app.js_ inside it. 

To render _index.html_, the function _hello()_ should return *render_template("index.html")*. 

{% highlight python %}
from flask import Flask, render_template
app = Flask(__name__,static_url_path="/static") 

@app.route("/") 
def hello(): 
    return render_template("index.html")
{% endhighlight %}

When you access _localhost:5001_, you will see the heading "A HTML File from templates", along with an alert that says "you are feeling app.js!". 

## Receiving URL encoded parameters

In order to process requests with parameters encoded in the url, we need to setup the corresponding route and the parameters are provided to a function that handles it.

{% highlight python %}
# GET
@app.route("/user/<username>") 
def name(username): 
    return "User %s " % username
{% endhighlight %}

The snippet above handles the url _/user/<username>_, where _username_ is a parameter passed by the user. This variable is passed to the _name()_ function which handles it. In this case, we just display the variable in the browser. Try it out for yourself. 

## Basic Authentication System

Sometimes we need to authenticate a user before serving him. Follow the code below carefully. 

{% highlight python %}
from flask import Flask, render_template, request, Response
from functools import wraps

app = Flask(__name__,static_url_path="/static") 

def check_auth(username, password):
    return username == 'admin' and password == 'secret'

def authenticate():
    # 401 response
    return Response('Could not verify your access level for that URL; \n You have to login with proper credentials',
            401,{'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    # f -> function being decorated
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args,**kwargs)
    return decorated

@app.route("/") 
@requires_auth
def secret_page(): 
    return render_template('index.html')


if (__name__ == "__main__"): 
    app.run(port = 5000) 

## Reference
# 1. http://flask.pocoo.org/snippets/8/
##
{% endhighlight %}




