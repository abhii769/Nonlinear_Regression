# Rational Model Fitting Using Gauss-Newton and Grid Search

This project implements a rational model fitting process using the Gauss-Newton optimization method and grid search. It includes functionality for estimating model parameters, confidence intervals, and residual analysis for a given dataset.

---

## Features

1. **Rational Model Implementation**:
   - Models the relationship between `t` (independent variable) and `y` (dependent variable) for three different type of linear and non-linear regression models
     

2. **Optimization Techniques**:
   - **Gauss-Newton Method**: Iteratively refines parameter estimates using gradient-based optimization.
   - **Grid Search**: Explores a range of initial values for parameters `b0` and `b1` to find the global minimum of the LSE.

3. **Confidence Intervals**:
   - Calculates 95% confidence intervals for the optimal parameters based on the Fisher Information Matrix (FIM).

4. **Visualization**:
   - Generates plots to visualize:
     - Residuals of the fitted model.
     - Observed data points and the fitted model curve.

---

## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries: `numpy`, `
