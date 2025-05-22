# Gradient Boosting Algorithm README

This document outlines the gradient boosting algorithm for supervised learning tasks, including regression and classification. The algorithm builds an ensemble of weak learners (typically decision trees) in a stage-wise fashion, optimizing a differentiable loss function.

## Algorithm Steps

### Input:
- Training set: $$ \{(x_i, y_i)\}_{i=1}^n $$
- Differentiable loss function: $$ L(y, F(x)) $$
- Number of iterations (boosting stages): $$ M $$

### 1. Initialization
Initialize the model with a constant value that minimizes the loss function:

$$  
f_0(x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \gamma)  
$$

### 2. Iterative Training (for $$ m = 1 $$ to $$ M $$):
#### (a) Compute Pseudo-Residuals
For each sample in the training set, compute the negative gradient (pseudo-residuals):

$$  
r_{im} = -\left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f=f_{m-1}}  
$$

#### (b) Fit a Regression Tree
Fit a regression tree to the pseudo-residuals $$ r_{im} $$, creating terminal regions $$ R_{jm} $$ for $$ j = 1, 2, \ldots, J_m $$.

#### (c) Compute Optimal Leaf Weights
For each terminal region $$ R_{jm} $$, compute the weight $$ \gamma_{jm} $$ that minimizes the loss:

$$  
\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)  
$$

#### (d) Update the Model
Update the current model by adding the new tree's predictions:

$$  
f_m(x) = f_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})  
$$

where $$ I(x \in R_{jm}) $$ is an indicator function for whether $$ x $$ belongs to region $$ R_{jm} $$.

### 3. Output the Final Model
After $$ M $$ iterations, the final model is:

$$  
\hat{f}(x) = f_M(x)  
$$

## Key Notes
- The algorithm generalizes to both regression and classification by choosing an appropriate loss function $$ L(y, F(x)) $$ (e.g., squared error for regression, log loss for classification).
- The step size (shrinkage) can be controlled by adding a learning rate $$ \nu $$ to the update step:  
  $$ f_m(x) = f_{m-1}(x) + \nu \cdot \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm}) $$.
- Early stopping or regularization techniques can be applied to prevent overfitting.
