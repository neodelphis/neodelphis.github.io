---
layout: post
mathjax: true
title:  "Propagation dans une couche convolutive"
date:   2019-07-02 12:00:00 +0200
categories:
  - convnet
  - python
---
Propagation dans une couche convolutive - draft

### Paramètres en entrée et sortie de la couche convolutive

A naive implementation of the forward pass for a convolutional layer.

The input consists of N data points, each with C channels, height H and width W. We convolve each input with F different filters, where each filter spans all C channels and has height HH and width WW.

Input:
- x: Input data of shape (N, C, H, W)
- w: Filter weights of shape (F, C, HH, WW)
- b: Biases, of shape (F,)
- conv_param: A dictionary with the following keys:
  - 'stride': The number of pixels between adjacent receptive fields in the
    horizontal and vertical directions.
  - 'pad': The number of pixels that will be used to zero-pad the input. 


During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides) along the height and width axes of the input. Be careful not to modfiy the original input x directly.

Returns a tuple of:
- out: Output data, of shape (N, F, H', W') where H' and W' are given by
  - H' = 1 + (H + 2 * pad - HH) / stride
  - W' = 1 + (W + 2 * pad - WW) / stride
- cache: (x, w, b, conv_param)


### Formulation mathématique  pour un filtre $y=f(x,w,b)$

Dimensions simplifiées
- x : $N \times N$
- $w$ : $m \times m$
- $\beta$ biais : scalaire
- y : $(N-m+1)\times (N-m+1)$

Propagation
$$y_{ij} = \left (\sum_{a=0}^{m-1} \sum_{b=0}^{m-1} \omega_{ab} x_{(i+a)(j+b)}  \right ) + \beta \tag {1}$$

### Cas particulier simple

Détail de la construction du produit de convolution simplifié avant généralisation

x


```python
import numpy as np
x = np.array([[[1, 2], [7, 4]],[[2, 3], [8, 3]],[[1, 1], [1, 1]]])
x.shape
```




    (3, 2, 2)



xp = x avec 0-padding de 1 sur chacun des canaux


```python
xp = np.pad(x,((0,), (1,), (1, )), 'constant')
xp
```




    array([[[0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 7, 4, 0],
            [0, 0, 0, 0]],
    
           [[0, 0, 0, 0],
            [0, 2, 3, 0],
            [0, 8, 3, 0],
            [0, 0, 0, 0]],
    
           [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]])



Définition d'un filtre w de taille 2x2 ( et toute la profondeur du volume d'entrée)


```python
w = np.array([[[0, 0],[0, 1]], [[0, 0],[0, 2]], [[0, 0],[0, 1]]])
w.shape
```




    (3, 2, 2)



Dimensions de la sortie:


```python
stride = 1
pad = 1
_, HH, WW = w.shape
H_ = int(1 + (H + 2 * pad - HH) / stride) # H'
W_ = int(1 + (W + 2 * pad - WW) / stride) # W'
H_, W_
```




    (3, 3)



Transformation en vecteur pour simplifier l'écriture du produit de convolution sous la forme d'un produit matriciel.


```python
w = w.reshape(-1)
w
```




    array([0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1])



Extraction du premier élément sur lequel va s'effectuer le produit de convolution:


```python
premier_elt = xp[:, 0:2, 0:2]
premier_elt
```




    array([[[0, 0],
            [0, 1]],
    
           [[0, 0],
            [0, 2]],
    
           [[0, 0],
            [0, 1]]])




```python
np.matmul(premier_elt.reshape(-1), w.T)
```




    6




```python
second_elt = xp[:, 1:3, 1:3]
np.matmul(second_elt.reshape(-1), w.T)
```




    11



Effet de la couche convolutive avec un filtre sans biais:


```python
y = np.zeros((H_, W_))
for i in range(H_):
    for j in range(W_):
        input_volume = xp[:, i*stride:i*stride+HH, j*stride:j*stride+WW]
        y[i,j] = np.matmul(input_volume.reshape(-1), w.T)
y
```




    array([[ 6.,  9.,  0.],
           [24., 11.,  0.],
           [ 0.,  0.,  0.]])



### Généralisation


```python
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # dimensions de la sortie (pas de tests sur la validité des choix)
    H_ = int(1 + (H + 2 * pad - HH) / stride)
    W_ = int(1 + (W + 2 * pad - WW) / stride)

    # 0-padding juste sur les deux dernières dimensions de x
    xp = np.pad(x, ((0,), (0,), (pad,), (pad, )), 'constant')
    
    out = np.zeros((N, F, H_, W_))
    
    for n in range(N):       # On parcourt toutes les images
        for f in range(F):   # On parcourt tous les filtres
            filter = w[f, :, :, :].reshape(-1)
            for i in range(H_):
                for j in range(W_):
                    input_volume = xp[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    out[n,f,i,j] = np.matmul(input_volume.reshape(-1), filter.T) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache
```
