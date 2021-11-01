#### Questions@:, Lecture 1 Questions
- Define the prediction function of a linear regression model and write down
  $L_2$-regularized mean squared error loss. [5]

- Starting from the unregularized sum of squares error of a linear regression model,
  show how the explicit solution can be obtained, assuming
  $\boldsymbol X^T \boldsymbol X$ is regular. [10]

#### Questions@:, Lecture 2 Questions
- Define expectation $\mathbb{E}[f(x)]$ and variance $\operatorname{Var}(f(x))$
  of a discrete random variable. Then define the bias of an estimator and show that
  estimating an expectation using a single sample is unbiased. [5]

- Describe standard gradient descent and compare it to stochastic (i.e., online) gradient
  descent and minibatch stochastic gradient descent. [5]

- Formulate conditions on the sequence of learning rates used in SGD to converge
  to optimum almost surely. [5]

- Write an $L_2$-regularized minibatch SGD algorithm for training a linear
  regression model, including the explicit formulas of the loss function and
  its gradient. [5]

#### Questions@:, Lecture 3 Questions
- Define binary classification, write down the perceptron algorithm and show
  how a prediction is made for a given example. [5]

- Show that the perceptron algorithm is an instance of stochastic gradient
  descent. Why are the learning rates not needed (i.e., why does not the result
  of the training depend on the learning rate)? [5]

- Define entropy, cross-entropy, Kullback-Leibler divergence, and prove
  the Gibbs inequality (i.e., that KL divergence is non-negative). [5]

- Define data distribution, empirical data distribution and likelihood. [5]

- Describe maximum likelihood estimation, as minimizing NLL, cross-entropy and
  KL divergence. [10]

- Considering binary logistic regression model, write down its parameters
  (including their size) and explain how is prediction performed (including the
  formula for the sigmoid function). Describe how we can interpret the outputs
  of the linear part of the model as logits. [5]

- Write down an $L_2$-regularized minibatch SGD algorithm for training a
  binary logistic regression model, including the explicit formulas of the
  loss function and its gradient. [10]

#### Questions@:, Lecture 4 Questions
- Define mean squared error and show how it can be derived using MLE. [5]

- Considering $K$-class logistic regression model, write down its parameters
  (including their size) and explain how is prediction performed (including the
  formula for the softmax function). Describe how we can interpret the outputs
  of the linear part of the model as logits. [5]

- Write down an $L_2$-regularized minibatch SGD algorithm for training
  a $K$-class logistic regression model, including the explicit formulas of the
  loss function and its gradient. [10]

- Considering a single-layer MLP with $D$ input neurons, $H$ hidden
  neurons, $K$ output neurons, hidden activation $f$ and output activation $a$,
  list its parameters (including their size) and write down how is the output
  computed. [5]

- List the definitions of frequently used MLP output layer activations (the ones
  producing parameters of a Bernoulli distribution and a categorical distribution).
  Then write down three commonly used hidden layer activations (sigmoid, tanh,
  ReLU). [5]

- Considering a single-layer MLP with $D$ input neurons, a ReLU hidden layer
  with $H$ units and softmax output layer with $K$ units, write down the
  formulas of the gradient of all the MLP parameters (two weight matrices and
  two bias vectors), assuming input $\boldsymbol x$, target $t$ and negative log
  likelihood loss. [10]

- Formulate the Universal approximation theorem. [5]

#### Questions@:, Lecture 5 Questions
- Consider derivation of softmax using maximum entropy principle, assuming
  we have a dataset of $N$ examples $(x_i, t_i), x_i \in \mathbb{R}^D,
  t_i \in \{1, 2, \ldots, K\}$. Formulate the three conditions we impose on the
  searched $\pi: \mathbb{R}^D \rightarrow \mathbb{R}^K$, and write down the
  Lagrangian to be maximized. [10]

- Define precision (including true positives and others), recall, $F_1$ score
  and $F_\beta$ score (we stated several formulations for $F_1$ and $F_\beta$
  scores; any of them will do). [5]

- Explain the difference between micro-averaged and macro-averaged $F_1$ score. [5]

- Describe k-nearest neighbors prediction, both for regression and
  classification. Define $L_p$ norm and describe uniform, inverse
  and softmax weighting. [5]

