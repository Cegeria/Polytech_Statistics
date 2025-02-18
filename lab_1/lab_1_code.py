import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm, poisson, uniform

#PLOTS

sample_sizes = [10, 50, 1000]

distributions = {
    "Cauchy": {
        "func": cauchy.rvs,
        "pdf": cauchy.pdf,
        "x_range": np.linspace(-10, 10, 1000)
    },
    "Normal": {
        "func": norm.rvs,
        "pdf": norm.pdf,
        "x_range": np.linspace(-5, 5, 1000)
    },
    "Poisson": {
        "func": lambda size: poisson.rvs(mu=10, size=size),
        "pdf": lambda x: poisson.pmf(x, mu=10),
        "x_range": np.arange(0, 10)
    },
    "Uniform": {
        "func": lambda size: uniform.rvs(loc=-np.sqrt(3), scale=2*np.sqrt(3), size=size),
        "pdf": lambda x: uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3)),
        "x_range": np.linspace(-2, 2, 1000)
    }
}

for dist_name, dist_info in distributions.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Distribution: {dist_name}")
    
    for i, size in enumerate(sample_sizes):
        samples = dist_info["func"](size=size)
        axes[i].hist(samples, bins='auto', density=True, alpha=0.6, color='g', label='Histogram')
        x = dist_info["x_range"]
        pdf = dist_info["pdf"](x)
        axes[i].plot(x, pdf, 'r-', label='PDF')
        
        axes[i].set_title(f"Sample size: {size}")
        axes[i].legend(loc='best', frameon=False)
        axes[i].set_xlim([x[0], x[-1]])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
    
    plt.show()

#CHARACTERISTICS

repeats = 1000

def calculate_statistics(samples):
    means = np.mean(samples, axis=1)
    medians = np.median(samples, axis=1)
    return means, medians

results = {}

for dist_name, dist_func in distributions.items():
    results[dist_name] = {}
    for size in sample_sizes:
        means_all = []
        medians_all = []
        z_q_all = []
        for _ in range(repeats):
            samples = dist_func["func"](size=size)
            first_quantils = np.quantile(samples, 0.25)
            medians = np.quantile(samples, 0.50)
            third_quantils = np.quantile(samples, 0.75)
            z_q = (first_quantils + third_quantils) / 2
            means = np.mean(samples)
            means_all.append(means)
            medians_all.append(medians)
            z_q_all.append(z_q)

        mean_of_means = np.mean(means_all)
        mean_of_medians = np.mean(medians_all)
        mean_of_z_q = np.mean(z_q_all)
        mean_of_squared_means = np.mean(np.square(means_all))
        mean_of_squared_medians = np.mean(np.square(medians_all))
        mean_of_squared_z_q = np.mean(np.square(z_q_all))
        dispersion = mean_of_squared_means - mean_of_means**2

        results[dist_name][size] = {
            "Mean": mean_of_means,
            "Median": mean_of_medians,
            "z_q": mean_of_z_q,
            "Mean square": mean_of_squared_means,
            "Median square": mean_of_squared_medians,
            "Z_q square": mean_of_squared_z_q,
            "Dispersion": dispersion
        }

for dist_name, sizes in results.items():
    print(f"Distribution: {dist_name}")
    for size, stats in sizes.items():
        print(f"  Sample size: {size}")
        for stat_name, value in stats.items():
            print(f"    {stat_name}: {value:.4f}")
    print()

