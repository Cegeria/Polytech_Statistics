import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
import pandas as pd
from matplotlib.cbook import boxplot_stats

sample_sizes = [20, 100, 1000]
distributions = {
    'Normal': lambda size: np.random.normal(0, 1, size),
    'Cauchy': lambda size: cauchy.rvs(0, 1, size),
    'Poisson': lambda size: np.random.poisson(10, size),
    'Uniform': lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size)
}

samples = {}
for name, func in distributions.items():
    samples[name] = {size: func(size) for size in sample_sizes}

#box-plot 
for dist_name in distributions:
    data = [samples[dist_name][size] for size in sample_sizes]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=sample_sizes, whis=1.5)
    plt.title(f'{dist_name} distribution')
    plt.xlabel('Sample size')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()

#ouliers 
outliers_count = {}
for dist_name in distributions:
    outliers_count[dist_name] = {}
    for size in sample_sizes:
        stats = boxplot_stats(samples[dist_name][size])
        outliers_count[dist_name][size] = len(stats[0]['fliers'])

df = pd.DataFrame(outliers_count).T
df.columns = [f'n={size}' for size in sample_sizes]
print("Outliers table:")
print(df)
