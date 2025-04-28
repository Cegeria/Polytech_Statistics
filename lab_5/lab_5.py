import numpy as np
from scipy.stats import norm, chi2

def chi2_normality_test(data, alpha=0.05, num_bins=10):
    n = len(data)
    mu_hat = np.mean(data)
    sigma_hat = np.std(data, ddof=0)
    
    if sigma_hat == 0:
        return "Стандартное отклонение равно 0, тест невозможен"
    
    percentiles = np.linspace(0, 1, num_bins + 1)
    quantiles = norm.ppf(percentiles, loc=mu_hat, scale=sigma_hat)
    
    observed, _ = np.histogram(data, bins=quantiles)
    expected = np.array([n / num_bins] * num_bins)
    
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = num_bins - 3  
    crit_value = chi2.ppf(1 - alpha, df)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    table = "Интервал\tНаблюдаемая\tОжидаемая\t(О-Е)^2/Е\n"
    for i in range(num_bins):
        bin_start = quantiles[i]
        bin_end = quantiles[i+1]
        obs = observed[i]
        exp = expected[i]
        contribution = (obs - exp)**2 / exp
        table += f"[{bin_start:.2f}, {bin_end:.2f})\t{obs}\t\t{exp:.1f}\t\t{contribution:.2f}\n"
    
    return chi2_stat, crit_value, p_value, table

np.random.seed(42)
data_normal = np.random.normal(size=100)
mu_hat = np.mean(data_normal)
sigma_hat = np.std(data_normal, ddof=0)
print(f"Оценки для нормальной выборки: μ = {mu_hat:.4f}, σ = {sigma_hat:.4f}\n")

chi2_stat, crit_val, p_val, table = chi2_normality_test(data_normal)
print(f"Хи-квадрат: {chi2_stat:.2f}, Критическое: {crit_val:.2f}, p-значение: {p_val:.4f}")
print("Таблица для нормальной выборки:")
print(table)

data_unif_100 = np.random.uniform(size=100)
chi2_stat_unif, crit_val_unif, p_val_unif, table_unif = chi2_normality_test(data_unif_100)
print(f"\nДля равномерной выборки 100: Хи-квадрат = {chi2_stat_unif:.2f}, p = {p_val_unif:.4f}")

data_unif_20 = np.random.uniform(size=20)
chi2_stat_20, crit_val_20, p_val_20, table_20 = chi2_normality_test(data_unif_20, num_bins=4)
print(f"\nДля равномерной выборки 20: Хи-квадрат = {chi2_stat_20:.2f}, p = {p_val_20:.4f}")
print("Таблица для выборки 20:")
print(table_20)
