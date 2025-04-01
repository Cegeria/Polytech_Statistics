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
    
    if num_first > 0:
        cov1 = [[1, 0.9], [0.9, 1]]
        samples1 = multivariate_normal.rvs(mean=[0, 0], cov=cov1, size=num_first)
    else:
        samples1 = np.empty((0, 2))
    
    if num_second > 0:
        cov2 = [[100, -90], [-90, 100]]
        samples2 = multivariate_normal.rvs(mean=[0, 0], cov=cov2, size=num_second)
    else:
        samples2 = np.empty((0, 2))
    
    samples = np.vstack([samples1, samples2])
    np.random.shuffle(samples)
    return samples[:, 0], samples[:, 1]

def compute_correlations(x, y):
    pearson_val, _ = pearsonr(x, y)
    spearman_val, _ = spearmanr(x, y)
    quadrant_val = quadrant_correlation(x, y)
    return pearson_val, spearman_val, quadrant_val

def plot_ellipse(ax, mean, cov, color, alpha=0.95):
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(lambda_)
    chi2_val = chi2.ppf(alpha, 2)
    width = 2 * np.sqrt(chi2_val) * lambda_[1]
    height = 2 * np.sqrt(chi2_val) * lambda_[0]
    angle = -np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linewidth=2)
    ax.add_patch(ellipse)

sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
n_iter = 1000

for size in sizes:
    for rho in rhos:
        pearson_list = []
        spearman_list = []
        quadrant_list = []
        for _ in range(n_iter):
            cov = [[1, rho], [rho, 1]]
            data = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=size)
            x, y = data[:, 0], data[:, 1]
            p, s, q = compute_correlations(x, y)
            pearson_list.append(p)
            spearman_list.append(s)
            quadrant_list.append(q)
        
        stats = {
            'Pearson': (np.mean(pearson_list), np.var(pearson_list)),
            'Spearman': (np.mean(spearman_list), np.var(spearman_list)),
            'Quadrant': (np.mean(quadrant_list), np.var(quadrant_list))
        }
        
        print(f"Normal: size={size}, rho={rho}")
        for key in stats:
            print(f"{key}: mean={stats[key][0]:.4f}, var={stats[key][1]:.4f}")
        print()

for size in sizes:
    pearson_list = []
    spearman_list = []
    quadrant_list = []
    for _ in range(n_iter):
        x, y = generate_mixture(size)
        p, s, q = compute_correlations(x, y)
        pearson_list.append(p)
        spearman_list.append(s)
        quadrant_list.append(q)
    
    stats = {
        'Pearson': (np.mean(pearson_list), np.var(pearson_list)),
        'Spearman': (np.mean(spearman_list), np.var(spearman_list)),
        'Quadrant': (np.mean(quadrant_list), np.var(quadrant_list))
    }
    
    print(f"Mixture: size={size}")
    for key in stats:
        print(f"{key}: mean={stats[key][0]:.4f}, var={stats[key][1]:.4f}")
    print()

def plot_samples_and_ellipses():
    for rho in rhos:
        cov = [[1, rho], [rho, 1]]
        data = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=100)
        x, y = data[:, 0], data[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, alpha=0.6)
        plot_ellipse(ax, [0, 0], cov, 'red', 0.95)
        plt.grid()
        ax.set_title(f'N(0,0,1,1,ρ={rho}) with 95% ellipse')
        plt.show()
    
    x, y = generate_mixture(100)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, alpha=0.6)
    plot_ellipse(ax, [0, 0], [[1, 0.9], [0.9, 1]], 'green', 0.95)
    plot_ellipse(ax, [0, 0], [[100, -90], [-90, 100]], 'blue', 0.95)
    plt.grid()
    ax.set_title('Mixture distribution with 95% ellipses')
    plt.show()

plot_samples_and_ellipses()
