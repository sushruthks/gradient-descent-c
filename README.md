# Gradient Descent in C

A minimal machine learning model in C that learns the relationship y = 2x from data using gradient descent, with every step written from scratch.

## What it does

The program is given five data points:

| x | y |
|---|---|
| 0 | 0 |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |

The hidden rule is `y = 2x`, but the program is never told that. It starts
from a wrong guess, `w = 3`, and trains the model `prediction = x * w` until
`w` is close to 2.

## How it works

**1. Cost function (mean squared error).** For a weight `w`, the program
measures how wrong the model is:

```
cost(w) = average of (y - x*w)^2 over all data points
```

Squaring keeps every error positive, so errors can't cancel each other out,
and it gives the cost a single lowest point at the best `w`.

| x | y | prediction (x*2) | error (y - prediction) | error^2 |
|---|---|------------------|------------------------|---------|
| 0 | 1 | 0                | +1                     | 1       |
| 1 | 1 | 2                | -1                     | 1       |
| 2 | 6 | 4                | +2                     | 4       |
| 3 | 4 | 6                | -2                     | 4       |
| 4 | 8 | 8                | 0                      | 0       |

**Without squaring:** the errors add up to +1 − 1 + 2 − 2 + 0 = **0**, so the
  model looks perfect, even though 4 of the 5 predictions are wrong.
**With squaring:** the cost is (1 + 1 + 4 + 4 + 0) / 5 = **2**, which
  correctly shows the model is still wrong.

**2. Slope of the cost (first-principles derivative).** To decide which way
to move `w`, the program nudges it by a tiny amount `dw` and measures how the
cost changes:

```
dcost/dw ≈ (cost(w + dw) - cost(w)) / dw
```

This is the first-principles definition of a derivative, with a small `dw`
(0.001) instead of a limit.

**3. Gradient descent update.** The weight takes a small step downhill:

```
w = w - rate * dcost/dw
```

If the slope is positive, `w` decreases; if it's negative, `w` increases.
The learning rate `rate` (0.01) controls the step size. Steps are big when
`w` is far from the answer and small when it's close.

## Build and run

You need a C compiler such as GCC.

```
gcc gradient_descent.c -o gradient_descent
./gradient_descent
```

On Windows, run `gradient_descent.exe`.

## Sample output

```
Training y = x * w on 5 data points

Step  0: w = 3.0000, cost = 6.000000
Step 10: w = 2.2782, cost = 0.464252
Step 20: w = 2.0771, cost = 0.035682
Step 30: w = 2.0211, cost = 0.002676
Step 40: w = 2.0055, cost = 0.000183
Step 50: w = 2.0012, cost = 0.000008

Learned weight: w = 2.0010 (the true rule is w = 2)
Prediction for x = 10: y = 20.0098
```


## Learning Outcomes

- How a model learns by minimising a cost function
- Why mean squared error squares the errors
- Application of Calculus in Machine Learning
- How gradient descent updates a parameter step by step
