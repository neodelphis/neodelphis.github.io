---
layout: post
mathjax: true
title:  "La fonction softmax et ses dérivées"
date:   2019-06-01 14:00:00 +0200
categories:
  - maths
---
$\partial Softmax$ : la fonction softmax et ses dérivées dans le cadre d'un réseau neuronal

# La fonction $Softmax$

## Définition

$$
\begin{align*}
Softmax, \forall N \in  \mathbb{N}^{*}\\
S\colon 
&\mathbb{R}^{N} &&\to \mathbb{R}^{N} \\
&a = 
\begin{bmatrix}
a_1\\ 
\vdots \\ 
a_n
\end{bmatrix}
&&\mapsto S(a) = 
\begin{bmatrix}
S_1\\ 
\vdots \\ 
S_n
\end{bmatrix}
\end{align*}
$$

Chaque élément de $S(a)$ est défini par: $$S_i = \frac{e^{a_i}}{ \sum_{k=1}^{N} e^{a_k} }$$  $$\forall i \in \{1,\cdots,N\}$$

Par souci de simplification d'écriture on a tendance à nommer S à la fois la fonction et le résultat de celle-ci, qui ici est un vecteur de taille N.

## Caractéristiques

$$\forall i, S_i \in ]0,1]$$

$$\sum_{i=1}^{N}S_i = 1$$

L'ordre des valeurs de `a` est conservé et la plus haute valeur ressort clairement. Par exemple pour `a = [1,0 2,0 5,0]`on aura `S(a) = [0,02 0,05 0,93]`

Softmax est comme une version "soft" de la fonction maximum, elle fait ressortir le maximum mais n'élimine pas complètement les autres valeurs.

## Instabilité numérique

$$
\frac{e^{\lambda_{y_i}}}{\sum_j e^{\lambda_j}}
= \frac{Ce^{\lambda_{y_i}}}{C\sum_j e^{\lambda_j}}
= \frac{e^{\lambda_{y_i} + \ln C}}{\sum_j e^{\lambda_j + \ln C}}
$$

Choix classique de $ln(C)$: valueur maximale du vecteur $\lambda$ $$\ln C = -\max_j \lambda_j$$


```python
import numpy as np
logits = np.array([123, 456, 789]) # Exemple de 3 classes avec de larges scores
p = np.exp(logits) / np.sum(np.exp(logits)) # Plante la machine
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in true_divide
      This is separate from the ipykernel package so we can avoid doing imports until



```python
# Translation du vecteur logits pour que la plus haute valeur soit 0:
logits -= np.max(logits) # logits devient [-666, -333, 0]
p = np.exp(logits) / np.sum(np.exp(logits)) # Fonctionne correctement
```


```python
def my_round(value, N):
    exponent = np.ceil(np.log10(value))
    return 10**exponent*np.round(value*10**(-exponent), N)
my_round(p,3)
```




    array([5.75e-290, 2.40e-145, 1.00e+000])



# Dérivée de $Softmax$

## Jacobien

Comme $Softmax$ est une fonction vectorielle, calculer sa dérivée correspond à déterminer sa matrice Jacobienne. Matrice de $i \in \{1,\cdots,N\}$ lignes et $j \in \{1,\cdots,N\}$ colonnes. Chaque élément est un scalaire défini par: $$\frac{\partial S_i}{\partial a_j}$$

La matrice Jacobienne est une matrice où chaque ligne contient l'ensemble des dérivées partielles de la composante $S_i$ par rapport à chacune des composantes $a_j$ du vecteur $a$.

Pour une écriture plus compacte on défini $$D_jS_i = \frac{\partial S_i}{\partial a_j}$$

$$DS = 
\begin{bmatrix}
D_1S_1 & \cdots  & D_NS_1\\ 
\vdots  & \ddots  & \vdots \\ 
D_1S_N & \cdots  & D_NS_N
\end{bmatrix}$$

## Dérivée d'une fraction de fonctions

Considérons la fonction f définie par $$f(x)= \frac{g(x)}{h(x)}$$

Sa dérivée selon x est donnée par la formule $$f'(x)= \frac{g'(x)h(x)-h'(x)g(x)}{h'(x)^2}$$

## Dérivée de S

$$
\begin{align*}
D_jS_i & = \frac{\partial S_i}{\partial a_j}\\
& = \frac{\partial }{\partial a_j}\frac{e^{a_i}}{ \sum_{k=1}^{N} e^{a_k} }
\end{align*}
$$

$S_i$ s'écrit sous la forme d'une fraction de deux éléments : $g_i = e^{a_i}$ et $h_i=\sum_{k=1}^{N} e^{a_k}$

En prenant les dérivées partielles par rapport à $a_j$ on obient:

$$
g'_i = 
\frac{\partial g_i}{\partial a_j} = 
   \begin{cases} 
   e^{a_i} & \text{lorsque } i=j \\
   0       & \text{sinon }
  \end{cases}
$$

Par rapport à $a_j$ tous les termes de la somme sont des constantes sauf $e^{a_j}$, donc

$$
h'_i = 
\frac{\partial h_i}{\partial a_j} = 
   e^{a_j} 
$$
- Lorsque $j = i$

$$
\begin{align*}
D_jS_i & = \frac{\partial S_i}{\partial a_j}\\
& = \frac{e^{a_i} \sum_{k=1}^{N} e^{a_k} - e^{a_i}e^{a_i}}{ (\sum_{k=1}^{N} e^{a_k})^2 }\\
& = \frac{e^{a_i}}{\sum_{k=1}^{N} e^{a_k} }  - \left (  \frac{e^{a_i}}{ \sum_{k=1}^{N} e^{a_k} }\right )^2\\
& = S_i - S_i^2\\
& = S_i(1 - S_i)
\end{align*}
$$

- Lorsque $j \neq i$

$$
\begin{align*}
D_jS_i & = \frac{\partial S_i}{\partial a_j}\\
& = \frac{0. h_i - e^{a_j}e^{a_i}}{ (\sum_{k=1}^{N} e^{a_k})^2 }\\
& = - S_j S_i\\
\end{align*}
$$

Donc on obtient l'équation finale pour la dérivée de S:
$$
D_jS_i = 
   \begin{cases} 
   S_i(1 - S_i) & \text{si } i=j \\
   - S_j S_i       & \text{si } i \neq j
  \end{cases}
$$

Ou en utilisant le symbole de Kronecker défini par:
$$
\delta_{ij} = 
   \begin{cases} 
   1 & \text{si } i=j \\
   0 & \text{si } i \neq j
  \end{cases}
$$

La dérivée de S peut s'écrire
$$
D_jS_i = S_i(\delta_{ij} - S_j) \\
\forall (i,j) \in \{1,\cdots,N\}^2 \\
$$

# Sources

[The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)


```python

```
