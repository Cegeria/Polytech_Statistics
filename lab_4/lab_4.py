import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
from math import ceil
import matplotlib.pyplot as plt

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
    x_avg = np.average(x)
    y_avg = np.average(y)
    x_2 = np.square(x)
    x_2_avg = np.average(x_2)
    xy = np.multiply(x, y)
    xy_avg = np.average(xy)

    beta_1 = (xy_avg - x_avg * y_avg) / (x_2_avg - x_avg * x_avg)
    beta_0 = y_avg - x_avg 
    return beta_0, beta_1

def lad_regression(x, y):
    x_l = ceil(x.size / 4)
    x_j = x.size - x_l + 1
    y_l = ceil(y.size / 4)
    y_j = y.size - y_l + 1

    med_x = np.median(x)
    med_y = np.median(y)
    sgn_x = np.sign(x - med_x)
    sgn_y = np.sign(y - med_y)
    n = len(x)
    r_q =  np.sum(sgn_x * sgn_y) / n

    #k_q = 1.349 - What is that???

    #q_y = (y[y_j-1] - y[y_l-1]) / k_q
    #q_x = (x[x_j-1] - x[x_l-1]) / k_q
    q_y = (y[y_j-1] - y[y_l-1]) 
    q_x = (x[x_j-1] - x[x_l-1]) 

    beta_1 = r_q * q_y / q_x
    beta_0 = med_y - beta_1 * med_x
    return beta_0, beta_1
    
a_ols, b_ols = ols_regression(x, y)
a_ols_2 = a_ols / 2
b_ols_2 = b_ols / 2
a_ols_pert, b_ols_pert = ols_regression(x, y_perturbed)
a_ols_pert_2 = a_ols_pert / 2
b_ols_pert_2 = b_ols_pert / 2

a_lad, b_lad = lad_regression(x, y)
a_lad_2 = a_lad / 2
b_lad_2 = b_lad / 2
a_lad_pert, b_lad_pert = lad_regression(x, y_perturbed)
a_lad_pert_2 = a_lad_pert / 2
b_lad_pert_2 = b_lad_pert / 2

print("Without perturbation:")
print(f"OLS: a^ = {a_ols:.2f}, a^/a = {a_ols_2:.2f}, b^ = {b_ols:.2f}, b^/b = {b_ols_2:.2f}")
print(f"LAD: a^ = {a_lad:.2f}, a^/a = {a_lad_2:.2f}, b = {b_lad:.2f}, b^/b = {b_lad_2:.2f}\n")

print("With perturbation:")
print(f"OLS: a^ = {a_ols_pert:.2f}, a^/a = {a_ols_pert_2:.2f}, b^ = {b_ols_pert:.2f}, b^/b = {b_ols_pert_2:.2f}")
print(f"LAD: a^ = {a_lad_pert:.2f}, a^/a = {a_lad_pert_2:.2f}, b = {b_lad_pert:.2f}, b^/b = {b_lad_pert_2:.2f}\n")

plt.figure(figsize=(10, 6))


plt.scatter(
    x, y, 
    color='blue', 
    marker='o', 
    label='Исходные точки',
    alpha=0.7
)

plt.plot(
    x, 2 + 2*x, 
    color='limegreen', 
    linewidth=3, 
    label='Эталон: $y=2+2x$'
)

plt.plot(
    x, a_ols + b_ols*x, 
    color='red', 
    linestyle='--', 
    label=f'МНК: $y={a_ols:.2f}+{b_ols:.2f}x$'
)

plt.plot(
    x, a_lad + b_lad*x, 
    color='darkviolet', 
    linestyle='-.', 
    label=f'МНМ: $y={a_lad:.2f}+{b_lad:.2f}x$'
)

plt.title('Регрессия без возмущений', fontsize=12)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(
    x, y_perturbed, 
    color='red', 
    marker='x', 
    label='Точки с возмущениями',
    alpha=0.7
)

plt.plot(
    x, 2 + 2*x, 
    color='limegreen', 
    linewidth=3, 
    label='Эталон: $y=2+2x$'
)

plt.plot(
    x, a_ols_pert + b_ols_pert*x, 
    color='red', 
    linestyle='--', 
    label=f'МНК: $y={a_ols_pert:.2f}+{b_ols_pert:.2f}x$'
)

plt.plot(
    x, a_lad_pert + b_lad_pert*x, 
    color='darkviolet', 
    linestyle='-.', 
    label=f'МНМ: $y={a_lad_pert:.2f}+{b_lad_pert:.2f}x$'
)

plt.title('Регрессия с возмущениями', fontsize=12)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
