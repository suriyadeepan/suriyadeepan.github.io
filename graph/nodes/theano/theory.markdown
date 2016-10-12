## scan

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
results, updates = theano.scan(fn=\_func, outputs\_info= [ { 'initial' : x0, 'taps' : [-2,-1] } ], n_steps = n)
fibo = theano.function([x0,n], results, updates=updates)
# call
fibo([0,1], 10)

```

## TODO

- [ ] Flow control
- [ ] Sequences
