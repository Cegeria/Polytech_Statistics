import numpy as np
from scipy import stats

def confidence_intervals_normal(x, alpha=0.05):
    n = len(x)
    mean = np.mean(x)
    s = np.std(x, ddof=1)  
    
    t_quantile = stats.t.ppf(1 - alpha/2, df=n-1)
    me_lower = mean - (s * t_quantile) / np.sqrt(n-1)
    me_upper = mean + (s * t_quantile) / np.sqrt(n-1)
    
    chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)
    sigma_lower = s * np.sqrt(n) / np.sqrt(chi2_upper)
    sigma_upper = s * np.sqrt(n) / np.sqrt(chi2_lower)
    
    return (me_lower, me_upper), (sigma_lower, sigma_upper)

def confidence_intervals_asymptotic(x, alpha=0.05):
    n = len(x)
    mean = np.mean(x)
    s = np.std(x, ddof=1)
    
    z_quantile = stats.norm.ppf(1 - alpha/2)
    me_lower = mean - (s * z_quantile) / np.sqrt(n)
    me_upper = mean + (s * z_quantile) / np.sqrt(n)
    
    m4 = np.mean((x - mean)**4)
    e = m4 / (s**4) - 3  
    U = z_quantile * np.sqrt((e + 2) / n)
    
    sigma_lower = s * (1 - 0.5 * U)
    sigma_upper = s * (1 + 0.5 * U)
    
    return (me_lower, me_upper), (sigma_lower, sigma_upper)

np.random.seed(42)  

n20_normal = np.random.normal(loc=0, scale=1, size=20)
n100_normal = np.random.normal(loc=0, scale=1, size=100)
n20_arbitrary = np.random.normal(loc=0, scale=0.5, size=20)  
n100_arbitrary = np.random.normal(loc=0, scale=0.5, size=100)

ci_n20_normal = confidence_intervals_normal(n20_normal)
ci_n100_normal = confidence_intervals_normal(n100_normal)

ci_n20_asym = confidence_intervals_asymptotic(n20_arbitrary)
ci_n100_asym = confidence_intervals_asymptotic(n100_arbitrary)

print("Доверительные интервалы для нормального распределения (n=20):")
print(f"m: [{ci_n20_normal[0][0]:.2f}, {ci_n20_normal[0][1]:.2f}]")
print(f"σ: [{ci_n20_normal[1][0]:.2f}, {ci_n20_normal[1][1]:.2f}]\n")

print("Доверительные интервалы для нормального распределения (n=100):")
print(f"m: [{ci_n100_normal[0][0]:.2f}, {ci_n100_normal[0][1]:.2f}]")
print(f"σ: [{ci_n100_normal[1][0]:.2f}, {ci_n100_normal[1][1]:.2f}]\n")

print("Доверительные интервалы для произвольного распределения (n=20) - асимптотический подход:")
print(f"m: [{ci_n20_asym[0][0]:.2f}, {ci_n20_asym[0][1]:.2f}]")
print(f"σ: [{ci_n20_asym[1][0]:.2f}, {ci_n20_asym[1][1]:.2f}]\n")

print("Доверительные интервалы для произвольного распределения (n=100) - асимптотический подход:")
print(f"m: [{ci_n100_asym[0][0]:.2f}, {ci_n100_asym[0][1]:.2f}]")
print(f"σ: [{ci_n100_asym[1][0]:.2f}, {ci_n100_asym[1][1]:.2f}]")
