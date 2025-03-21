import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate synthetic data from two Gaussian distributions
np.random.seed(42)
data = np.concatenate([np.random.normal(10, 2, 100), np.random.normal(20, 3, 100)])

# Step 2: Initialize parameters for two Gaussian distributions
mu1, mu2 = 8, 22  # Initial means
sigma1, sigma2 = 3, 3  # Initial standard deviations
pi1, pi2 = 0.5, 0.5  # Mixing coefficients

# Step 3: Expectation-Maximization Algorithm
def expectation_maximization(data, mu1, mu2, sigma1, sigma2, pi1, pi2, iterations=10):
    for _ in range(iterations):
        # E-Step: Compute responsibilities
        gamma1 = pi1 * norm.pdf(data, mu1, sigma1)
        gamma2 = pi2 * norm.pdf(data, mu2, sigma2)
        sum_gamma = gamma1 + gamma2
        gamma1 /= sum_gamma
        gamma2 /= sum_gamma
        
        # M-Step: Update parameters
        mu1 = np.sum(gamma1 * data) / np.sum(gamma1)
        mu2 = np.sum(gamma2 * data) / np.sum(gamma2)
        sigma1 = np.sqrt(np.sum(gamma1 * (data - mu1) ** 2) / np.sum(gamma1))
        sigma2 = np.sqrt(np.sum(gamma2 * (data - mu2) ** 2) / np.sum(gamma2))
        pi1 = np.mean(gamma1)
        pi2 = np.mean(gamma2)
    
    return mu1, mu2, sigma1, sigma2, pi1, pi2

# Run EM Algorithm
mu1, mu2, sigma1, sigma2, pi1, pi2 = expectation_maximization(data, mu1, mu2, sigma1, sigma2, pi1, pi2)

# Step 4: Print Final Parameters
print(f"Final Parameters:")
print(f"Cluster 1 -> Mean: {mu1:.2f}, Std Dev: {sigma1:.2f}, Mixing Coeff: {pi1:.2f}")
print(f"Cluster 2 -> Mean: {mu2:.2f}, Std Dev: {sigma2:.2f}, Mixing Coeff: {pi2:.2f}")

# Step 5: Visualization
x = np.linspace(min(data), max(data), 1000)
plt.hist(data, bins=30, density=True, alpha=0.6, color='gray')
plt.plot(x, pi1 * norm.pdf(x, mu1, sigma1), label=f'Gaussian 1 (μ={mu1:.2f}, σ={sigma1:.2f})', color='red')
plt.plot(x, pi2 * norm.pdf(x, mu2, sigma2), label=f'Gaussian 2 (μ={mu2:.2f}, σ={sigma2:.2f})', color='blue')
plt.title("Gaussian Mixture Model using Expectation-Maximization")
plt.legend()
plt.show()