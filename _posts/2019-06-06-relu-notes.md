---
layout: post
mathjax: true
title:  "Rétropropagation couche relu"
date:   2019-06-06 18:00:00 +0200
categories:
  - jupyter
  - maths
---
relu - notes

# Forward Pass


```python
# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.
import numpy as np

D = 4 # input_size
C = 3 # num_classes
#N = 5 # num_inputs

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(D) # un seul vecteur
    return X

X = init_toy_data()
np.around(X,3)
```




    array([ 16.243,  -6.118,  -5.282, -10.73 ])




```python
std=1e-1
W = std * np.random.randn(D, C)
b = np.zeros(C)
```


```python
np.around(W,3)
```




    array([[ 0.087, -0.23 ,  0.174],
           [-0.076,  0.032, -0.025],
           [ 0.146, -0.206, -0.032],
           [-0.038,  0.113, -0.11 ]])




```python
X  = np.hstack([X, np.array([1.])])
W  = np.vstack([W,b])
```


```python
np.around(X,3)
```




    array([ 16.243,  -6.118,  -5.282, -10.73 ,   1.   ])




```python
np.around(W,3)
```




    array([[ 0.087, -0.23 ,  0.174],
           [-0.076,  0.032, -0.025],
           [ 0.146, -0.206, -0.032],
           [-0.038,  0.113, -0.11 ],
           [ 0.   ,  0.   ,  0.   ]])




```python
Y  = np.zeros_like(X)
# logits
logits = np.matmul(X,W)
# relu
Y = np.maximum(logits, 0)
```


```python
np.around(logits,3)
```




    array([ 1.511, -4.062,  4.337])




```python
np.around(Y,3)
```




    array([1.511, 0.   , 4.337])



# Backpropagation in relu layer

$\lambda = logits$

On connaît $$\frac{\partial L}{\partial Y} = dL$$ où $L$ est le loss final. Dans la rétropropagation on va calculer successivement $$\frac{\partial L}{\partial \lambda} = \frac{\partial L}{\partial Y}.\frac{\partial Y}{\partial \lambda}$$ puis $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \lambda}.\frac{\partial \lambda}{\partial W}$$ et $$\frac{\partial L}{\partial X}$$


```python
# Par exemple on "rétro-passe":
dL = np.array([0.031, -0.415   , 0.926])
```

## $dY = \partial Y/\partial \lambda$

$$
\begin{align*}
\frac{\partial y_i}{\partial \lambda_j} & = \frac{\partial }{\partial \lambda_j}max(\lambda_i,0)\\
& = 
   \begin{cases} 
   1 & \text{ si } i=j \text{ et } \lambda_i>0 \\
   0       & \text{sinon }
  \end{cases}
\end{align*}
$$

dY est une matrice (C,C) avec seulement des termes sur la diagonale, non nuls lorsque $\lambda_i>0$

Ceci va nous permettre d'avoir une expression simplifiée de $\frac{\partial L}{\partial \lambda}$

## dlogits = $\partial L/\partial \lambda$

$$
\begin{align*}
\frac{\partial L}{\partial \lambda_k} & = \sum_{i}\frac{\partial L}{\partial y_i}.\frac{\partial y_i}{\partial \lambda_k}\\
& = 
   \begin{cases} 
   \frac{\partial L}{\partial y_k} & \text{si } \lambda_k>0 \\
   0       & \text{sinon }
  \end{cases}
\end{align*}
$$


```python
dlogits = dL*(logits>0)
dlogits
```




    array([ 0.031, -0.   ,  0.926])



## $\partial L/\partial W$

On a montré (cf BP-softmax-layer) que $$\frac{\partial L}{\partial W} = X^T.\frac{\partial L}{\partial \lambda}$$


```python
dW = np.outer(X, dL)
dW
```




    array([[ 0.50354706, -6.74103326, 15.04143807],
           [-0.18964449,  2.53878912, -5.66486439],
           [-0.16373324,  2.19191277, -4.89087043],
           [-0.33262027,  4.45281978, -9.93568944],
           [ 0.031     , -0.415     ,  0.926     ]])



# Version vectorisée avec N entrées X

## Forward pass


```python
import numpy as np

D = 4 # input_size
C = 3 # num_classes
N = 5 # num_inputs

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(N, D)
    return X

X = init_toy_data()

std=1e-1
W = std * np.random.randn(D, C)
b = np.zeros(C)
# b dans W
X  = np.hstack([X, np.ones((X.shape[0],1))])
W  = np.vstack([W,b])

print('X\n',np.around(X,3))
```

    X
     [[ 16.243  -6.118  -5.282 -10.73    1.   ]
     [  8.654 -23.015  17.448  -7.612   1.   ]
     [  3.19   -2.494  14.621 -20.601   1.   ]
     [ -3.224  -3.841  11.338 -10.999   1.   ]
     [ -1.724  -8.779   0.422   5.828   1.   ]]



```python
# Layer computation results
Y  = np.zeros_like(X)
# logits
logits = np.matmul(X,W)
# relu
Y = np.maximum(logits, 0)

print('logits\n',np.around(logits,3))
print('Y\n',np.around(Y,3))
```

    logits
     [[-2.599  2.545  2.45 ]
     [-2.727 -2.189  2.188]
     [-1.749  0.197  0.884]
     [-0.561 -1.015  0.105]
     [ 0.053 -1.431  0.202]]
    Y
     [[0.    2.545 2.45 ]
     [0.    0.    2.188]
     [0.    0.197 0.884]
     [0.    0.    0.105]
     [0.053 0.    0.202]]


## Backprop


```python
# Par exemple on "rétro-passe":
dL = std * (np.random.randn(N, C) - 0.5)
dL
```




    array([[ 0.00129298, -0.07980928, -0.00114819],
           [-0.05755717,  0.06316294,  0.10198168],
           [ 0.16855754, -0.18964963, -0.19441138],
           [-0.10044659, -0.03399629,  0.03761689],
           [-0.01843651, -0.25222012, -0.0806204 ]])




```python
dlogits = dL*(logits>0)
dlogits
```




    array([[ 0.        , -0.07980928, -0.00114819],
           [-0.        ,  0.        ,  0.10198168],
           [ 0.        , -0.18964963, -0.19441138],
           [-0.        , -0.        ,  0.03761689],
           [-0.01843651, -0.        , -0.0806204 ]])




```python
dW = X.T.dot(dlogits)
dW /= N
dW
```




    array([[ 0.00635795, -0.38028697,  0.05227749],
           [ 0.03236928,  0.19223368, -0.25841114],
           [-0.00155655, -0.47027045, -0.1329191 ],
           [-0.02149015,  0.95267558,  0.47151264],
           [-0.0036873 , -0.05389178, -0.02731628]])




```python
# b
dW[:, dW.shape[1]-1]
```




    array([ 0.05227749, -0.25841114, -0.1329191 ,  0.47151264, -0.02731628])


