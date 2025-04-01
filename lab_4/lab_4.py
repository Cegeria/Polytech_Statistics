import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

np.random.seed(123)  
x = np.linspace(-1.8, 2.0, 20)
epsilon = np.random.normal(0, 1, 20)
y = 2 + 2 * x + epsilon

y_perturbed = y.copy()
y_perturbed[0] += 10
y_perturbed[-1] -= 10

def objective(params, x, y):
    a, b = params
    residuals = y - (a + b * x)
    return np.sum(np.abs(residuals))

def ols_regression(x, y):
    x_avg = np.avg(x)
    y_avg = np.avg(y)
    x_2 = np.square(x)
    x_2_avg = mp.avg(x_2)
    xy = np.multiply(x, y)
    xy_avg = np.avg(xy)

    beta_1 = (xy_avg - x_avg * y_avg) / (x_2_avg - x_avg * x_avg)
    beta_0 = y_avg - x_avg 
    return beta_0, beta_1

def lad_regression(x, y):
    initial_guess = [0.0, 0.0]
    result = minimize(objective, initial_guess, args=(x, y), method='Powell')
    if result.success:
        return result.x
    else:
        raise ValueError("Fail")

a_ols, b_ols = ols_regression(x, y)
a_ols_pert, b_ols_pert = ols_regression(x, y_perturbed)

a_lad, b_lad = lad_regression(x, y)
a_lad_pert, b_lad_pert = lad_regression(x, y_perturbed)

print("Without perturbation:")
print(f"OLS: a = {a_ols:.3f}, b = {b_ols:.3f}")
print(f"LAD: a = {a_lad:.3f}, b = {b_lad:.3f}\n")

print("With perturbation:")
print(f"OLS: a = {a_ols_pert:.3f}, b = {b_ols_pert:.3f}")
print(f"LAD: a = {a_lad_pert:.3f}, b = {b_lad_pert:.3f}")
