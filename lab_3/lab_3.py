import numpy as np
from scipy.stats import multivariate_normal, pearsonr, spearmanr, chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def quadrant_correlation(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    product = (x - med_x) * (y - med_y)
    signs = np.sign(product)
    non_zero = signs != 0
    if np.sum(non_zero) == 0:
        return 0.0
    else:
        return np.mean(signs[non_zero])

def generate_mixture(n):
    indicators = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    num_first = np.sum(indicators == 0)
    num_second = n - num_first
    
    samples1 = np.empty((0, 2))
    if num_first > 0:
        cov1 = [[1, 0.9], [0.9, 1]]
        samples1 = multivariate_normal.rvs(mean=[0, 0], cov=cov1, size=num_first)
    
    samples2 = np.empty((0, 2))
    if num_second > 0:
        cov2 = [[100, -90], [-90, 100]]
        samples2 = multivariate_normal.rvs(mean=[0, 0], cov=cov2, size=num_second)
    
    samples = np.vstack([samples1, samples2])
    np.random.shuffle(samples)
    return samples[:, 0], samples[:, 1]

def compute_stats(data_list):
    mean = np.mean(data_list)
    mean_sq = np.mean(np.square(data_list))
    var = np.var(data_list)
    return mean, mean_sq, var

def plot_ellipse(ax, mean, cov, color, alpha=0.95):
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(lambda_)
    chi2_val = chi2.ppf(alpha, 2)
    width = 2 * np.sqrt(chi2_val) * lambda_[1]
    height = 2 * np.sqrt(chi2_val) * lambda_[0]
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=-angle,
                      edgecolor=color, facecolor='none', linewidth=2)
    ax.add_patch(ellipse)

# Основные параметры
sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
n_iter = 1000

# Обработка нормальных распределений
for size in sizes:
    for rho in rhos:
        stats_x = {'mean': [], 'mean_sq': [], 'var': []}
        stats_y = {'mean': [], 'mean_sq': [], 'var': []}
        stats_corr = {'Pearson': [], 'Spearman': [], 'Quadrant': []}
        for _ in range(n_iter):
            cov = [[1, rho], [rho, 1]]
            x, y = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=size).T
            
            # Статистики для x и y
            stats_x['mean'].append(np.mean(x))
            stats_x['mean_sq'].append(np.mean(np.square(x)))
            
            stats_y['mean'].append(np.mean(y))
            stats_y['mean_sq'].append(np.mean(np.square(y)))
            
            # Коэффициенты корреляции
            stats_corr['Pearson'].append(pearsonr(x, y)[0])
            stats_corr['Spearman'].append(spearmanr(x, y)[0])
            stats_corr['Quadrant'].append(quadrant_correlation(x, y))
        
        print(f"Normal: size={size}, rho={rho}")
        m_x = compute_stats(stats_x['mean'])[0]
        m_2_x = compute_stats(stats_x['mean_sq'])[0] 
        m_y = compute_stats(stats_y['mean'])[0]
        m_2_y = compute_stats(stats_y['mean_sq'])[0] 
        print(f"X: mean={m_x:.2f}, mean^2={m_2_x:.2f}")
        print(f"Y: mean={m_y:.2f}, mean^2={m_2_y:.2f}")
        
        p = compute_stats(stats_corr['Pearson'])[0]
        s = compute_stats(stats_corr['Spearman'])[0]
        q = compute_stats(stats_corr['Quadrant'])[0]

        print(f"Pearson: {p:.2f}, Spearman: {s:.2f}, Quadrant: {q:.2f}")
        print("\n")


for size in sizes:
    stats_x = {'mean': [], 'mean_sq': [], 'var': []}
    stats_y = {'mean': [], 'mean_sq': [], 'var': []}
    stats_corr = {'Pearson': [], 'Spearman': [], 'Quadrant': []}
    
    for _ in range(n_iter):
        x, y = generate_mixture(size)
        
        stats_x['mean'].append(np.mean(x))
        stats_x['mean_sq'].append(np.mean(np.square(x)))
        
        stats_y['mean'].append(np.mean(y))
        stats_y['mean_sq'].append(np.mean(np.square(y)))
        
        stats_corr['Pearson'].append(pearsonr(x, y)[0])
        stats_corr['Spearman'].append(spearmanr(x, y)[0])
        stats_corr['Quadrant'].append(quadrant_correlation(x, y))
    
    print(f"Mixture: size={size}")
    m_x = compute_stats(stats_x['mean'])[0]
    m_2_x = compute_stats(stats_x['mean_sq'])[0] 
    m_y = compute_stats(stats_y['mean'])[0]
    m_2_y = compute_stats(stats_y['mean_sq'])[0] 
    print(f"X: mean={m_x:.2f}, mean^2={m_2_x:.2f}")
    print(f"Y: mean={m_y:.2f}, mean^2={m_2_y:.2f}")
    
    p = compute_stats(stats_corr['Pearson'])[0]
    s = compute_stats(stats_corr['Spearman'])[0]
    q = compute_stats(stats_corr['Quadrant'])[0]

    print(f"Pearson: {p:.2f}, Spearman: {s:.2f}, Quadrant: {q:.2f}")
    print("\n")

# Визуализация (без изменений)
def plot_samples_and_ellipses():
    # Нормальные распределения
    for rho in rhos:
        cov = [[1, rho], [rho, 1]]
        data = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=100)
        x, y = data[:, 0], data[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, alpha=0.6)
        plot_ellipse(ax, [0, 0], cov, 'red', 0.95)
        ax.set_title(f'N(0,0,1,1,ρ={rho}) with 95% ellipse')
        plt.show()
    
    # Смесь
    x, y = generate_mixture(100)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, alpha=0.6)
    plot_ellipse(ax, [0, 0], [[1, 0.9], [0.9, 1]], 'green', 0.95)
    plot_ellipse(ax, [0, 0], [[100, -90], [-90, 100]], 'blue', 0.95)
    ax.set_title('Mixture distribution with 95% ellipses')
    plt.show()

plot_samples_and_ellipses()
