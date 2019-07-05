---
layout: post
mathjax: true
title:  "Rétro propagation dans une couche convolutive - Draft"
date:   2019-07-05 12:00:00 +0200
categories:
  - convnet
  - maths
  - python
---
# Rétro propagation dans une couche convolutive - Draft

## Notations utilisées

![conv layer diagram](/assets/images/conv-layer-diagram.jpg)

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


## Back prop in a convolutional layer

I struggled for quite a long time trying to find out how back propagation is working in a convolutional layer. As I was unable to find on the web a complete explanation of how it works. I decided to do the maths, trying to understand step by step how it's working on simple examples before generalizing.
The rest of the document is in french, but equations are self explanatory, and if there are requests I can post a translation.

### Forward
Dimensions avec un stride de 1 et pas de 0
- $x$ : $H \times W$
- $w$ : $HH \times WW$
- $b$ biais : scalaire
- $y$ : $H'\times W'$ 

Propagation - notations python
$$y_{ij} = \left (\sum_{k=0}^{HH-1} \sum_{l=0}^{WW-1} w_{kl} x_{i+k,j+l}  \right ) + b \tag {1}$$

### Rétro propagation

On connait $dy = \left(\frac{\partial L}{\partial y_{ij}}\right)$

On cherche $dx$, $d\omega$ et $d\beta$, dérivées partielles respectives de notre fonction de coût dont le gradient a été rétropropagé jusqu'à y.

# Cas d'un vecteur d'entrée x à 1 dimension

Pour essayer d'avoir une première intuition du résultat et voir comment les choses se mettent en place, on va se limiter dans un premier temps à un exemple très simple.

## En entrée

$$
x = 
\begin{bmatrix}
x_1\\ 
x_2\\ 
x_3\\ 
x_4
\end{bmatrix}
$$

$$
w = 
\begin{bmatrix}
w_1\\ 
w_2
\end{bmatrix}
$$

$$b$$

## En sortie

$$
y = 
\begin{bmatrix}
y_1\\ 
y_2\\ 
y_3
\end{bmatrix}
$$

## Propagation - convolution avec le filtre w, stride = 1, padding = 0

$$
y_1 = w_1 x_1 + w_2 x_2 + b\\
y_2 = w_1 x_2 + w_2 x_3 + b \tag{1}\\
y_3 = w_1 x_3 + w_2 x_4 + b
$$

## Rétropropagation

On connait le gradient de notre fonction de coût L par rapport à y:

$$
dy = \frac{\partial L}{\partial y}
$$

En fait $dy = \frac{\partial L}{\partial y}$, dérivée d'un scalaire par rapport à un vecteur s'écrit avec la notation du Jacobien:

$$
\begin{align*}
&
dy = 
\begin{bmatrix}
\frac{\partial L}{\partial y_1} & \frac{\partial L}{\partial y_2} & \frac{\partial L}{\partial y_3} 
\end{bmatrix} \\
&
dy = 
\begin{bmatrix}
dy_1 & dy_2 & dy_3
\end{bmatrix}
\end{align*}
$$

dy a les mêmes dimensions que y, écriture sous forme vectorielle:

$$
dy = (dy_1 , dy_2 , dy_3)
$$

On cherche

$$dx=\frac{\partial L}{\partial x},  dw=\frac{\partial L}{\partial w},  db=\frac{\partial L}{\partial b}$$

### db

$$db=\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial b} = dy\cdot\frac{\partial y}{\partial b}$$

Et la composée de fonction s'écrit sous la forme (en incorporant la formule de propagation (1)):

$$
db
=
\sum_{j}\frac{\partial L}{\partial y_j}\cdot \frac{\partial y_j}{\partial b} 
= 
\begin{bmatrix}
dy_1 & dy_2 & dy_3
\end{bmatrix}
\cdot
\begin{bmatrix}
1\\ 
1\\ 
1
\end{bmatrix}
$$

$$\Rightarrow db=dy_1+dy_2+dy_3$$

### dw

$$dw=\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial w} = dy\cdot\frac{\partial y}{\partial w}$$

$$
\frac{\partial y}{\partial w}
=
\begin{bmatrix}
\frac{\partial y_1}{\partial w_1} & \frac{\partial y_1}{\partial w_2}\\ 
\frac{\partial y_2}{\partial w_1} & \frac{\partial y_2}{\partial w_2}\\ 
\frac{\partial y_3}{\partial w_1} & \frac{\partial y_3}{\partial w_2}\\ 
\end{bmatrix}
$$

$$
\frac{\partial y}{\partial w}
=
\begin{bmatrix}
x_1 & x_2\\ 
x_2 & x_3\\ 
x_3 & x_4\\ 
\end{bmatrix}
$$

$$
\begin{bmatrix}
dy_1 & 
dy_2 & 
dy_3
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1 & x_2\\ 
x_2 & x_3\\ 
x_3 & x_4\\ 
\end{bmatrix}
$$

$$
dw_1 = x_1 dy_1 + x_2 dy_2 + x_3 dy_3 \\
dw_2 = x_2 dy_1 + x_3 dy_2 + x_4 dy_3
$$

dw correspond au produit de convolution de x avec dy comme filtre, à voir si cela se généralise avec une dimension supplémentaire.

$$
dw = 
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
*
\begin{bmatrix}
dy_1 \\
dy_2 \\
dy_3
\end{bmatrix}
$$

### dx

$$dx=\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial x} = dy^T\cdot\frac{\partial y}{\partial x}$$

$$
\frac{\partial y}{\partial x}
=
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \frac{\partial y_1}{\partial x_3} & \frac{\partial y_1}{\partial x_4}\\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \frac{\partial y_2}{\partial x_3} & \frac{\partial y_2}{\partial x_4}\\ 
\frac{\partial y_3}{\partial x_1} & \frac{\partial y_3}{\partial x_2} & \frac{\partial y_3}{\partial x_3} & \frac{\partial y_3}{\partial x_4}\\ 
\end{bmatrix}
$$

$$
\frac{\partial y}{\partial x}
=
\begin{bmatrix}
w_1 & w_2 & 0 & 0\\ 
0 & w_1 & w_2 & 0\\ 
0 & 0 & w_1 & w_2\\ 
\end{bmatrix}
$$

$$
\begin{align*}
&dx_1 = w_1 dy_1\\
&dx_2 = w_2 dy_1 + w_1 dy_2 \\
&dx_3 = w_2 dy_2 + w_1 dy_3 \\
&dx_4 = w_2 dy_3
\end{align*}
$$

Donne encore un produit de convolution, un peu particulier cette fois, il faudrait considérer dy avec un padding à 0 de 1, et en faire le produit convolutif avec un filtre w inversé du type $(w_2, w_1)$

$$
dx = 
\begin{bmatrix}
0 \\
dy_1 \\
dy_2 \\
dy_3 \\
0
\end{bmatrix}
*
\begin{bmatrix}
w_2 \\
w_1
\end{bmatrix}
$$

# Cas d'un vecteur d'entrée x à 2 dimensions

## En entrée

$$
x = 
\begin{bmatrix}
x_{11} &x_{12} &x_{13} &x_{14}\\ 
x_{21} &x_{22} &x_{23} &x_{24} \\ 
x_{31} &x_{32} &x_{33} &x_{34}\\ 
x_{41} &x_{42} &x_{43} &x_{44}
\end{bmatrix}
$$

$$
w = 
\begin{bmatrix}
w_{11} &w_{12}\\ 
w_{21} &w_{22}
\end{bmatrix}
$$

$$b$$

## En sortie

De nouveau on va prendre le cas le plus simple, stride de 1 et pas de padding. Donc $y$ aura pour dimension $3 \times 3$

$$
y = 
\begin{bmatrix}
y_{11} &y_{12} &y_{13} \\ 
y_{21} &y_{22} &y_{23} \\ 
y_{31} &y_{32} &y_{33}
\end{bmatrix}
$$

## Propagation

Ce qui nous donne:

$$
y_{11} = w_{11} x_{11} + w_{12} x_{12} + w_{21} x_{21} + w_{22} x_{22} + b\\
y_{12} = w_{11} x_{12} + w_{12} x_{13} + w_{21} x_{22} + w_{22} x_{23} + b\\
\cdots 
$$

En écriture indicielle:

$$y_{ij} = \left (\sum_{k=1}^{2} \sum_{l=1}^{2} w_{kl} x_{i+k-1,j+l-1}  \right ) + b \quad \forall(i,j)\in\{1,2,3\}^2 \tag {2}$$

## Rétropropagation

On connait:

$$
dy_{ij} = \frac{\partial L}{\partial y_{ij}}
$$

### db

En utilisant la convention d'Einstein pour alléger les notations (la répétition d'un indice indique la somme sur l'ensemble de la plage de valeurs de cet indice)

$$db = dy_{ij}\cdot\frac{\partial y_{ij}}{\partial b}$$

On a une double somme sur i et j, et $\forall (i,j)$ on a $\frac{\partial y_{ij}}{\partial b}=1$, donc

$$
db = \sum_{i=1}^3 \sum_{j=1}^3 dy_{ij}
$$

### dw

$$dw=\frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial w} = dy\cdot\frac{\partial y}{\partial w}$$

$$dw_{mn} = dy_{ij}\cdot\frac{\partial y_{ij}}{\partial w_{mn}} \tag{3}$$

On cherche

$$\frac{\partial y_{ij}}{\partial w_{mn}}$$

En incorporant l'équation (2) on obtient:

$$
\frac{\partial y_{ij}}{\partial w_{mn}}
= 
\sum_{k=1}^{2} \sum_{l=1}^{2} \frac{\partial w_{kl}}{\partial w_{mn}} x_{i+k-1,j+l-1}
$$

Tous les termes de $\frac{\partial w_{kl}}{\partial w_{mn}}$ sont nuls sauf pour $(k,l) = (m,n)$ où cela vaut 1, cas qui n'apparaît qu'une seule fois dans la double somme.

D'où:

$$
\frac{\partial y_{ij}}{\partial w_{mn}}
= 
x_{i+k-1,j+l-1}
$$

En remplaçant dans (3) on obtient:

$$dw_{mn} = dy_{ij} \cdot x_{i+k-1,j+l-1}$$

$$
\Rightarrow dw_{mn} = \sum_{i=1}^3 \sum_{j=1}^3 dy_{ij} \cdot x_{i+k-1,j+l-1}
$$

Si l'on compare cette équation avec l'équation 2 qui donne la formule d'un produit de convolution, on retrouve une structure similaire où dy joue le rôle de filtre que l'on applique sur x.

> - L'idée est d'appliquer ce filtre dy qui a une profondeur de 1, sur chacun des canaux de x et de sommer les valeurs pour obternir dw. Et aussi de faire la somme sur l'ensemble N des images utilisées.
- Dans la propagation, on fait une somme sur tous les canaux
- à détailler dans un chapitre ultérieur

### dx

En utilisant la loi de composition comme pour (3) on obtient:

$$
dx_{mn} = dy_{ij}\cdot\frac{\partial y_{ij}}{\partial x_{mn}} \tag{4}
$$

Cette fois ci on cherche 

$$
\frac{\partial y_{ij}}{\partial x_{mn}}
$$

En incorporant l'équation (2) on obtient:

$$\frac{\partial y_{ij}}{\partial x_{mn}}
= 
\sum_{k=1}^{2} \sum_{l=1}^{2} w_{kl} \frac{\partial x_{i+k-1,j+l-1}}{\partial x_{mn}}  \tag{5}
$$

On a 

$$
\frac{\partial x_{i+k,j+l}}{\partial x_{mn}}  = 
\begin{cases} 
1 & \text{si } m=i+k-1 \text{ et } n=j+l-1\\
0 & \text{sinon } 
\end{cases}
$$

$$
\begin{cases} 
m=i+k-1\\
n=j+l-1
\end{cases}
$$

$$
\Rightarrow
\begin{cases} 
k=m-i+1\\
l=n-j+1
\end{cases}
\tag{6}
$$

Dans notre exemple on a 

$$
\begin{align*}
&m,n \in [1,4] & \text{ entrées }\\
&k,l \in [1,2] & \text{ filtres }\\
&i,j \in [1,3] & \text{ sorties }
\end{align*}
$$

Donc lorsque l'on fait $k=m-i$, on va sortir un peu de l'intervalle de valeurs, $m-i+1 \in [-1,4]$
> - à voir comment on le gère
> - Ce changement d'indice correspond à ce qui nommé rot180 du filtre dans les papiers, à détailler?

De nouveau, dans la double somme de (5), on peut avoir une seule dérivée partielle de x qui soit égale à 1, lorsque l'on a (6), donc en remplaçant dans (5):

$$
\frac{\partial y_{ij}}{\partial x_{mn}}
= 
w_{m-i+1,n-j+1}
$$

où $w$ représente notre filtre initial étendu avec des valeurs 0, lorsque l'on sort de l'intervalle de définition

En injectant cette formule dans (4) on obtient:

$$
dx_{mn} = \sum_{i=1}^3 \sum_{j=1}^3 dy_{ij} \cdot w_{m-i+1,n-j+1} \tag{7}
$$

Par exemple 

$$
\begin{align*}
dx_{11} &= \sum_{i=1}^3 \sum_{j=1}^3 dy_{ij} \cdot w_{2-i,2-j}\\
&= \sum_{i=1}^3
dy_{i1} w_{2-i,1} +
dy_{i2} w_{2-i,0} +
dy_{i3} w_{2-i,-1,}\\
&= dy_{11} w_{1,1} + dy_{12} w_{1,0} + dy_{13} w_{1,-1}\\
&+ dy_{21} w_{0,1} + dy_{22} w_{0,0} + dy_{23} w_{0,-1}\\
&+ dy_{31} w_{-1,1} + dy_{32} w_{-1,0} + dy_{33} w_{-1,-1}
\end{align*}
$$

En utilisant $*$ pour notation du produit de convolution, on a:

$$
dx_{11} = dy * 
\begin{bmatrix}
w_{1,1} & 0 & 0 \\ 
0 & 0 & 0 \\ 
0 & 0 & 0
\end{bmatrix}
$$

Valeurs des indices de w pour $dx_{22}$ : $3-i,3-j$

$$
\begin{bmatrix}
2,2 & 2,1 & 2,0 \\ 
1,2 & 1,1 & 1,0 \\ 
0,2 & 0,1 & 0,0
\end{bmatrix}
$$

Donc on a un produit de convolution entre dy et une matrice w' de type:

$$
\begin{bmatrix}
w_{2,2} & w_{2,1} & 0 \\ 
w_{1,2} & w_{1,1} & 0 \\ 
0 & 0 & 0
\end{bmatrix}
$$

Autre exemple pour essayer de clarifier les choses: $dx_{43}$, de nouveau on se limite aux valeurs des indices: $4-i,3-j$

$$
\begin{bmatrix}
3,2 & 3,1 & 3,0 \\
2,2 & 2,1 & 2,0 \\
1,2 & 1,1 & 1,0 \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
w_{2,2} & w_{2,1} & 0 \\ 
w_{1,2} & w_{1,1} & 0
\end{bmatrix}
$$

Et du coup pour finir $dx_{44}$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & w_{2,2}
\end{bmatrix}
$$

Donc on voit bien apparaître un filtre w inversé, donc on obtient un produit convolutif entre $dy$ avec une bordure de 0 et $w'$, notre $w$ inversé, qui se déplace sur cette matrice avec un pas (stride) de 1.

$$w'_{ij}=w_{3-i,3-j}$$

$$dx = dy\_0 * w' \tag{8}$$ 

## Résumé des équations de rétropropagation

$$
db = \sum_{i=1}^3 \sum_{j=1}^3 dy_{ij}
$$

$$
dw = 
\begin{bmatrix}
x_{11} &x_{12} &x_{13} &x_{14}\\ 
x_{21} &x_{22} &x_{23} &x_{24} \\ 
x_{31} &x_{32} &x_{33} &x_{34}\\ 
x_{41} &x_{42} &x_{43} &x_{44}
\end{bmatrix}
*
\begin{bmatrix}
dy_{11} &dy_{12} &dy_{13} \\ 
dy_{21} &dy_{22} &dy_{23} \\ 
dy_{31} &dy_{32} &dy_{33}
\end{bmatrix}
= x * dy
$$

$$
dx = 
\begin{bmatrix}
0 &0 &0 &0 &0 \\
0 &dy_{11} &dy_{12} &dy_{13} &0\\ 
0 &dy_{21} &dy_{22} &dy_{23} &0\\ 
0 &dy_{31} &dy_{32} &dy_{33} &0\\
0 &0 &0 &0 &0
\end{bmatrix}
*
\begin{bmatrix}
w_{22} & w_{21} \\ 
w_{12} & w_{11} \\ 
\end{bmatrix}
= dy\_0 * w'
$$

