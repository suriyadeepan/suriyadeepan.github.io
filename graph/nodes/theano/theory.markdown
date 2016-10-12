---
layout: page
title: Theano
---

## scan

The **scan** function helps us create loops natively, instead of using python's loops. At each iteration, a function is called. It can also be used to iterate through a tensor. The code snippet below demonstrates the use of scan, for generating fibonacci sequence.


```python
import theano
import theano.tensor as T

# Fibonacci Sequence
#  inputs
x0 = T.ivector('x0')
n = T.iscalar('n')
# fibonacci step
_func = lambda a,b : a+b
# scan
results, updates = theano.scan(fn=_func, outputs_info= [ { 'initial' : x0, 'taps' : [-2,-1] } ], n_steps = n)
fibo = theano.function([x0,n], results, updates=updates)
# call
fibo([0,1], 50)

```

Let us try to incorporate flow control in scan. Instead of using *_func*, let us use *step* function, defined below.

```python
def step(a,b):
	return a+b, theano.scan_module.until(a+b < 0)

results, updates = theano.scan(fn=_func, outputs_info= [ { 'initial' : x0, 'taps' : [-2,-1] } ], n_steps = n)
fibo = theano.function([x0,n], results, updates=updates)
# call
fibo([0,1], 50)
```

*theano.scan_module.until* provides us, a means to stop the loop given a condition, which in this case is, stop the loop when the result is negative.


## References

* [Understanding scan() in Theano](http://nbviewer.jupyter.org/gist/triangleinequality/1350873eebea33973e41)


## TODO

- [ ] Newton's Method
- [ ] Sequences













