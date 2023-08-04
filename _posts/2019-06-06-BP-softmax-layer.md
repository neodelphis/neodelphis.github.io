---
mathjax: true
title:  "Rétropropagation du gradient softmax"
date:   2019-06-06 10:06:00 +0200
categories:
  - maths
---

Rétropropagation du gradient dans le cadre d'une ultime couche $Softmax$

Cet article présente la rétropropagation du gradient dans un réseau neuronal monocouche complètement connecté, avec softmax comme fonction d'activation et la divergence de Kullback-Leibler comme fonction de coût. Minimiser cette fonction qui fait office de distance entre nos deux distributions va se résumer dans notre cas à réduire l'entropie croisée entre le résultat obtenu et le résultat souhaité.

Le but est d'optimiser une matrice de poids W pour que la prédiction d'appartenance de X à une classe soit la plus proche de la classe Y connue.

## Présentation du contexte

### Graphe

<br>
Ce que l'on connait:<br>
$X$ vecteur d'entrées de dimension (D)<br>
$y$ classe à laquelle appartient le vecteur d'entrées, $y$ est un scalaire $\in \{1,\cdots,C\}$. On associe à $y$ un vecteur de dimension C: $$Y\_one\_hot$$, qui est la version "one hot encoded" de $y$. Tous ses termes sont 0 sauf $$Y\_one\_hot[y]=1$$<br>
<br>
Ce que l'on cherche:<br>
$W$ matrice des poids de dimension (D,C). Où C représente le nombre de classes possibles<br>
<br>
Représentation générale du graphe des calculs effectués:<br>

** 1 ** multiplication matricielle : $\lambda = X.W$<br>

$\lambda$ vecteur logits de dimension (C)<br>

** 2 ** $S=softmax(\lambda)$<br>

$S$ vecteur de dimension (C) qui donne la probabilité d'appartenance de X pour chacune des classes<br>

** 3 ** Fonction de coût: $$L = D_{KL}(Y\_one\_hot\|S)$$ où $$Y\_one\_hot$$ correspond à la répartition de probabilité pour la classe connue.<br>

$L$ (Loss) scalaire (1)


### Motivation

Le but est de trouver $$\frac{\partial L}{\partial W}$$ Jacobien généralisé de L par rapport à W, pour pouvoir modifier W en utilisant le taux d'apprentissage:

$$ W \leftarrow W + learning\_rate * \frac{\partial L}{\partial W}$$

### La fonction $Softmax$

#### Définition

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

Chaque élément de $S(a)$ est défini par: 

$$S_i = \frac{e^{a_i}}{ \sum_{k=1}^{N} e^{a_k} }$$  $$\forall i \in \{1,\cdots,N\}$$

Par souci de simplification d'écriture on a tendance à nommer de la même manière une fonction et son résultat. $S$ ici est selon le contexte la fonction $softmax$ ou un vecteur de taille N.

#### Caractéristiques

$$\forall i, S_i \in ]0,1]$$

$$\sum_{i=1}^{N}S_i = 1$$

L'ordre des valeurs de `a` est conservé et la plus haute valeur ressort clairement. Par exemple pour `a = [1,0 2,0 5,0]`on aura `S(a) = [0,02 0,05 0,93]`

Softmax est comme une version "soft" de la fonction maximum, elle fait ressortir le maximum mais n'élimine pas complètement les autres valeurs.

#### Instabilité numérique

Le problème du calcul de la valeur de $softmax$ est que l'on divise facilement de très grands nombres entre eux, source d'instabilités numériques. Pour éviter cela on va ajouter une constante judicieusement choisie.

$$
\frac{e^{\lambda_{i}}}{\sum_j e^{\lambda_j}}
= \frac{Ce^{\lambda_{i}}}{C\sum_j e^{\lambda_j}}
= \frac{e^{\lambda_{i} + \ln C}}{\sum_j e^{\lambda_j + \ln C}}
$$

Choix classique de $ln(C)$: valeur maximale du vecteur $\lambda$ 

$$\ln C = -\max_j \lambda_j$$


```python
import numpy as np
logits = np.array([123, 456, 789]) ## Exemple de 3 classes avec de larges scores
p = np.exp(logits) / np.sum(np.exp(logits)) ## Plante la machine
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in true_divide
      This is separate from the ipykernel package so we can avoid doing imports until



```python
## Translation du vecteur logits pour que la plus haute valeur soit 0:
logits -= np.max(logits) ## logits devient [-666, -333, 0]
p = np.exp(logits) / np.sum(np.exp(logits)) ## Fonctionne correctement
```


```python
def my_round(value, N):
    exponent = np.ceil(np.log10(value))
    return 10**exponent*np.round(value*10**(-exponent), N)
my_round(p,3)
```




    array([5.75e-290, 2.40e-145, 1.00e+000])



Dans notre cas de figure on calculera numériquement $$S=softmax(\hat{\lambda})=softmax(\lambda)$$ où $\hat{\lambda}$ est une version translatée de $\lambda$ pour éviter les instabilités numériques mais qui est égale à $S$. $\hat{\lambda}$ vecteur logits translaté de dimension (C)

## Gradient

On utilise la loi donnant la dérivée d'une composée de fonctions:

$$
\frac{\partial L}{\partial W} =   \frac{\partial L}{\partial \lambda} . \frac{\partial \lambda}{\partial W}\\
$$

Ce qui nous donne en considérant les dimensions: $$(D,C) =  (C) . (C,D,C)$$

où le "." correspond au produit matriciel généralisé aux tenseurs.

### Jacobien généralisé de $\lambda$ par rapport à $W$

$$\lambda = X.W$$

$$J = \frac{\partial \lambda}{\partial W}$$

Dimensions: X(D) - W(D,C) - $lambda (C)$ - J(C,D,C)

$$J_{ijk} = \frac{\partial \lambda_i}{\partial w_{jk}}$$

$$\text{où (i,j,k) varient selon (C,D,C)} $$

$$\lambda_i=\sum_{l=1}^{D}x_lw_{li}$$

$$
\begin{align*}
J_{ijk} & = \frac{\partial}{\partial w_{jk}}\sum_{l=1}^{D}x_lw_{li}\\
& = 
   \begin{cases} 
   x_j & \text{si } i=k, \forall j \in[1,D] \\
   0       & \text{si } i \neq k
  \end{cases}
\end{align*}
$$

### Jacobien de $L$ par rapport à $W$

$$
\frac{\partial L}{\partial W} =   \frac{\partial L}{\partial \lambda} . \frac{\partial \lambda}{\partial W}\\
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_{jk}}  & = \sum_{i=1}^{C}\frac{\partial L}{\partial \lambda_i}J_{ijk} \\
& =  \frac{\partial L}{\partial \lambda_k}x_j
\end{align*}
$$

Donc sous forme matricielle on obtient: $$
\frac{\partial L}{\partial W} = X^T.\frac{\partial L}{\partial \lambda}$$

### Jacobien de L par rapport à X

De même on peut monter: 
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \lambda}.W^T
$$

### Expression de la dérivée de $L$ par rapport à $\lambda$

#### Fonction de coût

$$L = D_{KL}(Y\_one\_hot\|S)$$ où $$Y\_one\_hot$$ correspond à la répartition de probabilité pour la classe connue et S les probabilités de la classe prédite par notre modèle.

La divergence de Kullback-Leibler $D_{KL}$ correspond à une mesure de la similarité entre deux distributions. On peut l'écrire sous la forme $$D_{KL} = H(p,q)-H(p)$$ où p est la véritable distribution et q la répartition estimée. 

$$
\begin{align*}
&p_i = Y\_one\_hot_i = 
   \begin{cases} 
   1 & \text{si } i=y  \\
   0 & \text{si } i \neq y \forall i \in[1,C]
  \end{cases} \\
&q_i = S(\lambda)_i
\end{align*}
$$

Comme 100% des valeurs de p sont en y on a $H(p)=O$, d'où:

$$
\begin{align*}
D_{KL}  & = H(p,q) \\
& =  -\sum_{i}p_i lnq_i
\end{align*}
$$

Tous les termes de la somme sont nuls sauf pour $i=y$

$$
\begin{align*}
L = D_{KL} & = - ln \frac{e^{\lambda_{y}}}{\sum_j e^{\lambda_j}} \\
& =  -\lambda_{y} + ln(\sum_j e^{\lambda_j})
\end{align*}
$$

#### Calcul de la dérivée de L

Calcul préliminaire: 

$$
\begin{align*}
\frac{\partial}{\partial \lambda_i}ln(\sum_j e^{\lambda_j}) & = \frac{e^{\lambda_{i}}}{\sum_j e^{\lambda_j}} \\
& =  S_i
\end{align*}
$$

On obtient une expression simple en fonction de $softmax(\lambda)$

$$
\frac{\partial L}{\partial \lambda_i} = 
   \begin{cases} 
   S_y - 1 & \text{si } i=y  \\
   S_i & \text{si } i \neq y  , \forall i \in[1,C]
  \end{cases}
$$  

### Gradients dW et dX

En incorporant cette dernière formule à dW, Jacobien de $L$ par rapport à $W$, on obtient une expression du gradient facilement programmable. 

$$
\begin{align*}
&\frac{\partial L}{\partial W} = X^T.\frac{\partial L}{\partial \lambda} \\
&\frac{\partial L}{\partial \lambda_i} = 
   \begin{cases} 
   S_y - 1 & \text{si } i=y  \\
   S_i & \text{si } i \neq y , \forall i \in[1,C]
  \end{cases}
\end{align*}
$$

Sur [github](https://github.com/neodelphis/cs231n-assignment1/blob/master/cs231n/classifiers/softmax.py) une version avec boucles et une version vectorisée de cette fonction.

De même, le couple d'équations pour exprimer dX, Jacobien de L par rapport à X:

$$
\begin{align*}
&\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \lambda}.W^T \\
&\frac{\partial L}{\partial \lambda_i} = 
   \begin{cases} 
   S_y - 1 & \text{si } i=y  \\
   S_i & \text{si } i \neq y \forall i \in[1,C]
  \end{cases}
\end{align*}
$$ 

### Références

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/linear-classify/#softmax)

[Backpropagation for a Linear Layer - Justin Johnson](http://vision.stanford.edu/teaching/cs231n/handouts/linear-backprop.pdf)

[Demystifying KL Divergence](https://towardsdatascience.com/demystifying-kl-divergence-7ebe4317ee68)

[How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
